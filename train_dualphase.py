###########
import os
seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)
import keras
from keras import layers
import numpy as np
from cv2 import threshold
import random
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard

from Modules.custom_layers.custom_pretrained import get_custom_pretrained
from Modules.custom_layers.attention import *
from Modules.Models.FPN_dual_hal import FPNModelDual as FPNModelDual
#from Modules.Models.FPN_certainty import FPNModel as FPNModelDistil
from Modules.Models.FPN_distil import FPNModel as FPNModelDistil
from Modules.custom_layers.blocks import *
from Modules.Models.FPNRAW import FPNModelRAW
from Modules.Models.UNET import Unet
from Modules.Models.DeepLabv3 import DeepLabV3Plus
from Modules.Models.UNETattn import UnetAttn
from sklearn.utils.class_weight import compute_class_weight

from import_config_kickoff import load_config

#from dataloader_dual_certainty import DataGenerator, train_augment, test_val_augment
from dataloader_dual import DataGenerator, train_augment, test_val_augment


from utils import *
from loss_new import *
#from distiller_dual_phase_certainty import *
from distiller_dual_phase import *

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_Mean_IoU', mode='max', verbose=1, keep_only_latest=True):
        """
        Custom callback to save model weights at the end of the epoch if the metric improves.
        Uses TensorFlow Checkpoint format for saving weights only.

        Arguments:
            - filepath: directory or base file path to save the checkpoint.
            - monitor: metric to monitor.
            - mode: 'max' or 'min'.
            - verbose: 0 or 1.
            - keep_only_latest: whether to overwrite the previous checkpoint.
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.keep_only_latest = keep_only_latest
        self.best = -np.Inf if mode == 'max' else np.Inf
        self.last_saved_path = None
        self.checkpoint = None

    def on_epoch_end(self, epoch, logs=None):
        if self.model is None or self.model.student is None:
            raise ValueError("Model or student attribute is not set in the callback.")

        if self.checkpoint is None:
            self.checkpoint = tf.train.Checkpoint(model=self.model.student)

        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (self.mode == 'max' and current > self.best) or (self.mode == 'min' and current < self.best)
        if improved:
            self.best = current
            filename = os.path.join(self.filepath, f"best_checkpoint")  # no phase, no epoch number
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.4f}, saving checkpoint to {filename}")

            # Save model weights in TensorFlow's format (folder of files)
            self.checkpoint.write(filename)  # Overwrites the previous checkpoint in-place

            # Update the last saved path
            self.last_saved_path = filename

# Select classes for binary segmentation
selected_classes = ['Non-Mining', 'Mining']

metrics = [
        AverageF1Score(threshold=0.5, name='Average_F1_Score'),
        AveragePrecision(threshold=0.5, name='Average_Precision'),
        AverageRecall(threshold=0.5, name='Average_Recall'),
        MeanIOU(threshold=0.5, name='Mean_IoU'),
        'accuracy'
    ]

height = 256
width = 256
num_classes = 1
num_channel = 6    
mod = 'FPN' #FPN, FPNRAW, UNET, DeepLabV3+

def get_model(BACKBONE='densenet121', num_classes=num_classes, channel=num_channel, mod='FPN', trainable=True, 
              class_weights=None):
    base_model, layers = get_custom_pretrained(
        name=BACKBONE, weights='imagenet', channels=channel, 
        include_top=False, pooling=None, width=width, height=height
    )

    if mod == 'FPN':
        model = FPNModelDistil(base_model=base_model, height=height, width=width, channel=channel,
                               output_layers=layers, classes=num_classes, backbone_trainable=trainable)
    # other models...

    loss_model = CombinedLoss(class_weights=class_weights)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=opt, loss=loss_model, metrics=metrics)

    dummy_input = tf.zeros((1, height, width, channel))
    
    # Only call model with the image input, no need for other dummy inputs
    model(dummy_input, training=False)

    return model


class WarmUpThenReduce(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, base_lr, reduce_lr_callback):
        super(WarmUpThenReduce, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.reduce_lr_callback = reduce_lr_callback
        self.epoch = 0

    def set_model(self, model):
        self.model = model
        # Make sure the inner callback also gets the model
        if self.reduce_lr_callback is not None:
            self.reduce_lr_callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            warmup_lr = self.base_lr * ((epoch + 1) / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.lr, warmup_lr)
            print(f"\n[WarmUp] Epoch {epoch + 1}: Setting learning rate to {warmup_lr:.6f}")
        else:
            # Allow ReduceLROnPlateau to take over
            pass

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.warmup_epochs:
            self.reduce_lr_callback.on_epoch_end(epoch, logs)


def reset_random_seeds(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

def train_model(args):
    configs_wb = dict(
        num_classes = int(args.n_classes),
        batch_size = int(args.batch_size),
        learning_rate = float(args.lr),
        epochs = int(args.n_epochs),
        backbone = args.backbone,
        seed = int(args.seed)
    )

    config = load_config(args)
    reset_random_seeds(seed_value=seed_value)
    seed = configs_wb["seed"]
    batch_size = configs_wb["batch_size"]
    BACKBONE = configs_wb["backbone"]
    num_classes = configs_wb["num_classes"]

    # Define training and validation path
    train_img_path = config["train_img_dir"]
    train_mask_path = config["train_mask_dir"]

    val_img_path = config["val_img_dir"]
    val_mask_path = config["val_mask_dir"]
    
    num_train_images = len(os.listdir(train_img_path))
    num_val_images = len(os.listdir(val_img_path))

    step_per_epoch = np.ceil(num_train_images / batch_size)  # Use ceil to ensure all images are used
    val_step_per_epoch = np.ceil(num_val_images / 1)  # For validation, assumes batch_size=1

    # Build teacher model with wrapper
    teacher_model = get_model(
        BACKBONE=BACKBONE,
        mod=mod,
        trainable=True,
    )

    # Load weights into the inner FPN model
    teacher_model.load_weights(config["teacher_model_weights"], by_name=True, skip_mismatch=True)

    # Freeze everything
    teacher_model.trainable = False



    if args.useaug:
        train_image_gen = DataGenerator(
            image_path=train_img_path,
            fine_mask_path=train_mask_path,
            teacher_model=teacher_model,  # Updated here
            batch_size=batch_size,
            classes=num_classes,
            selected_classes=selected_classes,
            height=height,
            width=width,
            channel=int(args.num_channel),
            augmentation=train_augment(),
            use_aug=True,
            shuffle=True,
            seed=seed
        )
    else:
        train_image_gen = DataGenerator(
            image_path=train_img_path,
            fine_mask_path=train_mask_path,
            teacher_model=teacher_model,  # Updated here too
            batch_size=batch_size,
            classes=num_classes,
            selected_classes=selected_classes,
            height=height,
            width=width,
            channel=int(args.num_channel),
            augmentation=test_val_augment(),
            use_aug=False,
            shuffle=True,
            seed=seed
        )

    val_image_gen = DataGenerator(
        image_path=val_img_path,
        fine_mask_path=val_mask_path,
        teacher_model=teacher_model,  # Updated here as well
        batch_size=1,
        classes=num_classes,
        selected_classes=selected_classes,
        height=height,
        width=width,
        channel=int(args.num_channel),
        augmentation=test_val_augment(),
        use_aug=False,
        shuffle=True,
        seed=seed
    )


    # LR scheduler
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_Mean_IoU',
        factor=0.5,
        patience=3,
        cooldown=1,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )

    warmup_then_reduce = WarmUpThenReduce(
        warmup_epochs=5, base_lr=0.001, reduce_lr_callback=lr_scheduler)

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_Mean_IoU',
    patience=10,
    mode='max',
    restore_best_weights=True,
    verbose=1
    )

    checkpoint = CustomModelCheckpoint(
        filepath=config["save_weight"] + args.name_file_weight + '_' + args.channel + '_' + args.model + '_' + BACKBONE + "_best" + ".weights.h5",
        monitor='val_Mean_IoU',
        mode='max',
        verbose=1
    )

    csvlogger = CSVLogger(
        filename=config["save_csv"] + args.name_file_csv + '_' + args.channel + '_' + args.model + '_' + BACKBONE + ".csv",
        separator="\t",
        append=False
    )

    base_model, layers = get_custom_pretrained(name=BACKBONE, weights='imagenet', width=width, height=height, channels=int(args.num_channel), include_top=False, pooling=None)
    
    if args.model == "FPN":
        print("**********************************")
        print(f"Use FPN Model with {BACKBONE} for {args.channel} Images")
        model = FPNModelDual(base_model=base_model, output_layers=layers, classes=num_classes, height=height, width=width, channel=int(args.num_channel), 
                final_activation=args.activation_output, backbone_trainable=args.set_trainable)
    else:
        raise ValueError("Select one of the following models: FPN, FPNRAW, UNET, DeepLabV3+")
    
    if args.class_weight == "yes":
        # class_weights = np.array([1.0, 4.62])  # For Non-Mining: 1.22, Mining: 5.63
        class_weights = np.array([0.24, 4.19])  # Extreme weight: heavily prioritize Mining class
        print("With Class Weights")
        print("**********************************")
    else:
        class_weights = None
        print("No Class Weights")
        print("**********************************")


    # === Loss & Optimizer ===
    loss_model = CombinedLossBoundarySSIM(class_weights=class_weights)
    
    opt = tf.keras.optimizers.experimental.AdamW(
        learning_rate=configs_wb["learning_rate"],
        weight_decay=1e-4
    )

    callbacks = [checkpoint, csvlogger, warmup_then_reduce, early_stopping]

    distiller = Distiller(student=model, teacher=teacher_model)
    distiller.compile(
        optimizer=opt,
        metrics=metrics,
        student_loss_fn=loss_model,
        distillation_loss_fn=keras.losses.KLDivergence(),
        contrastive_loss_fn=pixel_wise_contrastive_loss,
        alpha=1,
        temperature=5,
        contrastive_weight=1.0,
        boundary_loss_fn=BoundaryLoss(),
    )

    distiller.fit(
        train_image_gen,
        steps_per_epoch=step_per_epoch,
        epochs=configs_wb["epochs"],
        callbacks=callbacks,
        verbose=1,
        validation_data=val_image_gen,
        validation_steps=val_step_per_epoch
    ) 


    # Define the directory path to save last epoch weights
    last_epoch_dir = os.path.join(
        config["save_weight"], 
        f"{args.name_file_weight}_{args.channel}_{args.model}_{BACKBONE}_last_epoch"
    )

    # Ensure the directory exists
    os.makedirs(last_epoch_dir, exist_ok=True)

    # File prefix inside the directory (TensorFlow will add .index and .data files)
    filename = os.path.join(last_epoch_dir, "ckpt")

    # Create a checkpoint object
    checkpoint = tf.train.Checkpoint(model=distiller.student)

    # Save the checkpoint
    checkpoint.write(filename)

    print(f"\nLast epoch: Weights saved to {last_epoch_dir}")

