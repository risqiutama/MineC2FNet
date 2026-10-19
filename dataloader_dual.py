import tifffile as tiff
import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

# Classes for binary segmentation
CLASSES = ['Non-Mining', 'Mining']  # Non-Mining = 0, Mining = 255

#######################AUGMENTATION######################
def train_augment():
    transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        ], p=0.5),

        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # optional but helps robustness

        A.Normalize(mean=0.0, std=1.0, always_apply=True)
    ])
    return transform


def test_val_augment():
    transform = A.Compose([
        A.Normalize(mean=0.0, std=1.0, always_apply=True)
        #A.Lambda(mask=round_clip_0_1)
    ])
    return transform

############################
# Get image and label path
def get_image_label_path(images_dir, masks_dir, shuffle=False, seed=None):
    images_ids = sorted(os.listdir(images_dir))
    mask_ids = sorted(os.listdir(masks_dir))

    if shuffle:
        if seed is not None:
            tf.random.set_seed(seed)

        indices = tf.range(start=0, limit=tf.shape(images_ids)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        # images_ids = tf.gather(images_ids, shuffled_indices).numpy().astype(str)
        # mask_ids = tf.gather(mask_ids, shuffled_indices).numpy().astype(str)
        images_ids = [id.decode('utf-8') for id in tf.gather(images_ids, shuffled_indices).numpy()]
        mask_ids = [id.decode('utf-8') for id in tf.gather(mask_ids, shuffled_indices).numpy()]


    images_files = [os.path.join(images_dir, i) for i in images_ids]
    masks_files = [os.path.join(masks_dir, i) for i in mask_ids]

    while True:
        for i in range(len(images_files)):
            yield [images_files[i], masks_files[i]]

def pad_image_and_mask(image, mask, target_height, target_width):
    # Resize image and mask to target size
    image_resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # Ensure mask is single channel (binary mask)
    if mask_resized.ndim == 3 and mask_resized.shape[2] > 1:
        mask_resized = mask_resized[..., 0]  # Convert to single channel

    # Add an additional channel to the mask (for compatibility with the model input)
    mask_resized = mask_resized[..., np.newaxis]  # shape (target_height, target_width, 1)

    return image_resized, mask_resized

def preprocess_image_label(image_file, mask_file, selected_classes=None, target_height=None, target_width=None, augmentation=None):
    # Read the image and mask
    img = tiff.imread(image_file)  # Image (typically, you load in RGB or RGBNIR)
    mask = cv2.imread(mask_file, 0).astype('float32')  # Read the mask as grayscale

    # If the mask has multiple channels, we convert it to binary (0 and 1)
    if selected_classes is None:
        mask = np.where(mask == 255, 1.0, 0.0)  # Convert to binary mask (0, 1)
        mask = mask[..., np.newaxis]  # Add channel dimension to mask (height, width, 1)
    else:
        if len(selected_classes) == 1 and 'Mining' in selected_classes and 'Non-Mining' in selected_classes:
            # Check if the mask already contains only 0s and 1s
            if np.array_equal(np.unique(mask), [0, 1]):
                pass  # The mask is already in 0 and 1, do nothing
            else:
                mask = np.where(mask == 255, 1.0, 0.0)  # Convert Mining (255) to 1.0, Non-Mining (0) remains 0.0
            mask = mask[..., np.newaxis]  # Add channel dimension to mask (height, width, 1)


    # Resize the image and mask to the target size
    if target_height and target_width:
        img, mask = pad_image_and_mask(img, mask, target_height, target_width)

    # Apply augmentation if provided
    if augmentation:
        sample = augmentation(image=img, mask=mask)
        img, mask = sample['image'], sample['mask']

    return img, mask

def DataGenerator(image_path, fine_mask_path, teacher_model, batch_size, classes, selected_classes,
                  height, width, channel, augmentation, use_aug=False, shuffle=True, seed=None):
    images_labels = get_image_label_path(images_dir=image_path, masks_dir=fine_mask_path,
                                         shuffle=shuffle, seed=seed)

    # Dummy pass to get coarse_pred + feature shapes
    dummy_input = tf.zeros((1, height, width, channel))  # Dummy input for feature shape determination
    dummy_pred, dummy_feats = teacher_model(dummy_input, training=False, return_features=True)

    # Get feature shapes for the teacher model's feature maps
    feature_shapes = [feat.shape[1:] for feat in dummy_feats]

    while True:
        images = np.zeros((batch_size, height, width, channel), dtype=np.float32)

        #print("Student input shape:", images.shape)
        fine_masks = np.zeros((batch_size, height, width, classes), dtype=np.float32)
        coarse_preds = np.zeros((batch_size, height, width, classes), dtype=np.float32)
        coarse_feats = [np.zeros((batch_size, *shape), dtype=np.float32) for shape in feature_shapes]

        for i in range(0, batch_size, 2):
            image_file, fine_mask_file = next(images_labels)

            # Augmented version
            image_aug, fine_label_aug = preprocess_image_label(
                image_file, fine_mask_file,
                selected_classes=selected_classes,
                target_height=height, target_width=width,
                augmentation=augmentation if use_aug else test_val_augment()
            )

            # Get coarse predictions and features from teacher model (only pass the input image)
            coarse_pred_aug, feats_aug = teacher_model(np.expand_dims(image_aug, axis=0), training=False, return_features=True)
            feats_aug = [f[0] for f in feats_aug]  # Remove batch dim

            images[i] = image_aug
            fine_masks[i] = fine_label_aug
            coarse_preds[i] = coarse_pred_aug[0]  # Remove batch dim
            for j in range(len(coarse_feats)):
                coarse_feats[j][i] = feats_aug[j]

            # Non-augmented version
            if i + 1 < batch_size:
                image_no_aug, fine_label_no_aug = preprocess_image_label(
                    image_file, fine_mask_file,
                    selected_classes=selected_classes,
                    target_height=height, target_width=width,
                    augmentation=test_val_augment()
                )

                # Get coarse predictions and features from teacher model (only pass the input image)
                coarse_pred_no_aug, feats_no_aug = teacher_model(np.expand_dims(image_no_aug, axis=0), 
                                                                  training=False, return_features=True)
                feats_no_aug = [f[0] for f in feats_no_aug]

                images[i + 1] = image_no_aug
                fine_masks[i + 1] = fine_label_no_aug
                coarse_preds[i + 1] = coarse_pred_no_aug[0]
                for j in range(len(coarse_feats)):
                    coarse_feats[j][i + 1] = feats_no_aug[j]

        # Yield the data in the format the student model expects
        yield ((tf.convert_to_tensor(images),
                tf.convert_to_tensor(coarse_preds),
                [tf.convert_to_tensor(f, dtype=tf.float32) for f in coarse_feats]),
            tf.convert_to_tensor(fine_masks, dtype=tf.float32))
