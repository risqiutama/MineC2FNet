#############


import os
seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)

import numpy as np
from cv2 import threshold
import random

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard

from Modules.custom_layers.custom_pretrained import get_custom_pretrained
from Modules.Models.FPN import FPNModel
from Modules.Models.FPNRAW import FPNModelRAW
from Modules.Models.UNET import Unet
from Modules.Models.DeepLabv3 import DeepLabV3Plus
from Modules.Models.UNETattn import UnetAttn
from sklearn.utils.class_weight import compute_class_weight

from import_config_kickoff import load_config

from dataloader import DataGenerator, train_augment, test_val_augment

from utils import *
from loss_new import *

# Select classes for binary segmentation
selected_classes = ['Non-Mining', 'Mining']


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

    height = 256
    width = 256
    
    step_per_epoch = np.ceil(num_train_images / batch_size)  # Use ceil to ensure all images are used
    val_step_per_epoch = np.ceil(num_val_images / 1)  # For validation, assumes batch_size=1

    if args.useaug == True:
        train_image_gen = DataGenerator(train_img_path, train_mask_path, height=height, width=width, selected_classes=selected_classes, 
                                    channel=int(args.num_channel), augmentation=train_augment(),
                                    classes=num_classes, batch_size=batch_size, shuffle=True, seed=seed)
    else:
        train_image_gen = DataGenerator(train_img_path, train_mask_path, height=height, width=width, selected_classes=selected_classes, 
                                channel=int(args.num_channel), augmentation=test_val_augment(),
                                classes=num_classes, batch_size=batch_size, shuffle=True, seed=seed)

    val_image_gen = DataGenerator(val_img_path, val_mask_path, height=height, width=width, selected_classes=selected_classes, 
                                channel=int(args.num_channel), augmentation=test_val_augment(),
                                classes=num_classes, batch_size=1, shuffle=True, seed=seed)

    exponential_decay_fn = exponential_decay(float(args.lr), 60)

    # lr_scheduler = LearningRateScheduler(
    #     exponential_decay_fn,
    #     verbose=1
    # )


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
    patience=6,
    mode='max',
    restore_best_weights=True,
    verbose=1
    )





    checkpoint = ModelCheckpoint(
        filepath = config["save_weight"] + args.name_file_weight + '_' + args.channel + '_' + args.model + '_' + BACKBONE + "_best" + ".weights.h5",
        save_best_only=True, 
        save_weights_only=True,
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
        model = FPNModel(base_model=base_model, output_layers=layers, classes=num_classes, height=height, width=width, channel=int(args.num_channel), 
                final_activation=args.activation_output, backbone_trainable=args.set_trainable)
    elif args.model == "FPNRAW":
        print("**********************************")
        print(f"Use FPN RAW Model for {args.channel} Images")
        model = FPNModelRAW(classes=num_classes, height=height, width=width, channel=int(args.num_channel), final_activation=args.activation_output)
    elif args.model == "UNET":
        print("**********************************")
        print(f"Use UNET Model with {BACKBONE} for {args.channel} Images")
        model = Unet(base_model=base_model, output_layers=layers, classes=num_classes, height=height, width=width, channel=int(args.num_channel),
                final_activation=args.activation_output, backbone_trainable=args.set_trainable)
    elif args.model == "DeepLabV3+":
        print("**********************************")
        print(f"Use DeepLabV3+ Model with {BACKBONE} for {args.channel} Images")
        model = DeepLabV3Plus(base_model=base_model, output_layers=layers, classes=num_classes, height=height, width=width, channel=int(args.num_channel), 
                              output_stride=16, final_activation=args.activation_output, backbone_trainable=args.set_trainable)
    elif args.model == "flood":
        print("**********************************")
        print(f"Use flood Model with {BACKBONE} for {args.channel} Images")
        base_model_nir, layers_nir = get_custom_pretrained(name=BACKBONE, weights='imagenet', width=width, height=height, channels=1, include_top=False, pooling=None)

        model = UnetAttn(base_model=base_model, output_layers=layers, base_model_nir=base_model_nir, output_layers_nir=layers_nir, classes=num_classes, height=height, width=width, channel=int(args.num_channel),
                final_activation=args.activation_output, backbone_trainable=args.set_trainable)
    else:
        raise ValueError("Select one of the following models: FPN, FPNRAW, UNET, DeepLabV3+")
    
    if args.class_weight == "yes":
        # class_weights = np.array([1.0, 4.62])  # For Non-Mining: 1.22, Mining: 5.63
        class_weights = np.array([1.22, 5.63])  # Extreme weight: heavily prioritize Mining class
        print("With Class Weights")
        print("**********************************")
    else:
        class_weights = None
        print("No Class Weights")
        print("**********************************")

    loss_model = CombinedLossBoundarySSIM(class_weights=class_weights)

    metrics = [
        AverageF1Score(threshold=0.5, name='Average_F1_Score'),
        AveragePrecision(threshold=0.5, name='Average_Precision'),
        AverageRecall(threshold=0.5, name='Average_Recall'),
        MeanIOU(threshold=0.5, name='Mean_IoU'),
        'accuracy'
    ]

    opt = tf.keras.optimizers.Adam(learning_rate=configs_wb["learning_rate"])
    
    model.compile(optimizer=opt, loss=loss_model, metrics=metrics)
    
    callbacks = [checkpoint, csvlogger]
    #callbacks = [checkpoint, csvlogger, warmup_then_reduce, early_stopping]


    model.fit(
        train_image_gen,
        steps_per_epoch=step_per_epoch,
        epochs=configs_wb["epochs"],
        callbacks=callbacks,
        verbose=1,
        validation_data=val_image_gen,
        validation_steps=val_step_per_epoch
    )

    #Save model
    model.save_weights(config["save_weight"] + args.name_file_weight + '_' + args.channel + '_' + args.model + '_' + BACKBONE + ".hdf5", save_format='h5')
    print(f"Training Finish! Best model and Final model weights saved to Saved_Weights")