import tensorflow as tf
from utils import *
import cv2
import numpy as np


def dice_score(y_true, y_pred, smooth=1e-7, class_weights=1, class_indexes=None, threshold=None):
    print("y_true shape:", tf.shape(y_true))
    print("y_pred shape:", tf.shape(y_pred))
    print("class_weights shape:", tf.shape(class_weights))
    y_true, y_pred = gather_channels(y_true, y_pred, indexes=class_indexes)
    y_pred = round_if_needed(y_pred, threshold)
    #axes = [1, 2]  # Spatial dimensions
    axes = range(1, K.ndim(y_pred))

    numerator = y_true * y_pred * class_weights  # Broadcasting
    numerator = 2. * K.sum(numerator, axis=axes)

    denominator = (y_true + y_pred) * class_weights # Broadcasting
    denominator = K.sum(denominator, axis=axes)

    score = (numerator + smooth) / (denominator + smooth)

    score = K.mean(score)

    return score

def iou_score(y_true, y_pred, smooth=1e-7, class_weights=1, class_indexes=None, threshold=0.5):    
    y_true, y_pred = gather_channels(y_true, y_pred, indexes=class_indexes)

    # Apply thresholding to the predictions
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Convert probabilities to binary masks

    axes = range(1, K.ndim(y_pred))  # Apply along spatial dimensions

    # IoU calculation
    numerator = K.sum(y_true * y_pred * class_weights, axis=axes) 
    denominator = K.sum((y_true + y_pred) * class_weights, axis=axes) - numerator

    score = (numerator + smooth) / (denominator + smooth)
    score = K.mean(score)

    return score

def precision(y_true, y_pred, smooth=1e-7, class_weights=1, class_indexes=None, threshold=None):    
    y_true, y_pred = gather_channels(y_true, y_pred, indexes=class_indexes)
    y_pred = round_if_needed(y_pred, threshold)
    #axes = [1, 2]
    axes = range(1, K.ndim(y_pred))

    #y_true = K.flatten(y_true)
    #y_pred = K.flatten(y_pred)

    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp

    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, class_weights)

    return score


def recall(y_true, y_pred, smooth=1e-7, class_weights=1, class_indexes=None, threshold=None):    
    y_true, y_pred = gather_channels(y_true, y_pred, indexes=class_indexes)
    y_pred = round_if_needed(y_pred, threshold)
    #axes = [1, 2]
    axes = range(1, K.ndim(y_pred))

    #y_true = K.flatten(y_true)
    #y_pred = K.flatten(y_pred)

    tp = K.sum(y_true * y_pred, axis=axes)
    fn = K.sum(y_true, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, class_weights)

    return score


def categorical_crossentropy(y_true, y_pred, class_indexes=None, class_weights=1):
    class_weights = check_class_weights(class_weights)
    y_true, y_pred = gather_channels(y_true, y_pred, indexes=class_indexes)

    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())  

    cce_loss = y_true * K.log(y_pred) * class_weights

    axis_to_reduce = range(1, K.ndim(cce_loss))
    cce_loss = K.mean(cce_loss, axis=axis_to_reduce)

    return -cce_loss



class DiceLoss(Loss):
    def __init__(self, smooth=1e-7, class_weights=1, class_indexes=None, threshold=None):
        super().__init__(name="dice_loss")
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth
        self.threshold=threshold
        self.class_indexes = class_indexes

    def __call__(self, y_true, y_pred):
        return 1.0 - dice_score(
            y_true,
            y_pred,
            smooth=self.smooth,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            threshold=None
        )


class CCELoss(Loss):
    def __init__(self, class_weights=1, class_indexes=None):
        super().__init__(name="categorical_crossentropy")
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.class_indexes = class_indexes

    def __call__(self, y_true, y_pred):
        return categorical_crossentropy(
            y_true,
            y_pred,
            class_weights = self.class_weights,
            class_indexes=self.class_indexes      
        )


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, class_weights=None, threshold=None):
        super().__init__(name="binary_crossentropy")
        
        # Handle the case where class_weights is None, defaulting to [1.0, 1.0]
        if class_weights is None:
            self.class_weights = tf.constant([1.0, 1.0], dtype=tf.float32)
        else:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        # Clip predictions to prevent log(0) errors
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Apply threshold if provided (useful if predictions need to be binarized)
        if self.threshold is not None:
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        # Calculate binary cross-entropy loss with weights
        loss = - (self.class_weights[1] * y_true * tf.math.log(y_pred) +
                  self.class_weights[0] * (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Return mean loss across the batch
        return tf.reduce_mean(loss)



class IOUScore(Metric):
    def __init__(self, threshold=0.5, smooth=1e-7, class_weights=1, class_indexes=None, name=None):
        name = name or "iou_score"
        super().__init__(name=name)
        self.threshold = threshold
        self.smooth = smooth
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.class_indexes = class_indexes

    def __call__(self, y_true, y_pred):
        return iou_score(
            y_true,
            y_pred,
            smooth=self.smooth,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            threshold=self.threshold
        )


class FScore(Metric):
    def __init__(self, threshold=None, class_weights=1, class_indexes=None, smooth=1e-7, name=None):
        name = name or "f1_score"
        super().__init__(name=name)
        self.threshold = threshold
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.class_indexes = class_indexes
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return dice_score(
            y_true,
            y_pred,
            smooth=self.smooth,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            threshold=self.threshold
        )

class Precision(Metric):
    def __init__(self, threshold=None, class_weights=1, class_indexes=None, smooth=1e-7, name=None):
        name = name or "precision_score"
        super().__init__(name=name)
        self.threshold = threshold
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.class_indexes = class_indexes
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return precision(
            y_true,
            y_pred,
            smooth=self.smooth,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            threshold=self.threshold
        )
    
class Recall(Metric):
    def __init__(self, threshold=None, class_weights=1, class_indexes=None, smooth=1e-7, name=None):
        name = name or "recall_score"
        super().__init__(name=name)
        self.threshold = threshold
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.class_indexes = class_indexes
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return recall(
            y_true,
            y_pred,
            smooth=self.smooth,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            threshold=self.threshold
        )

class AveragePrecision(Metric):
    def __init__(self, threshold=0.5, smooth=1e-7, class_weights=1, name="average_precision"):
        super().__init__(name=name)
        self.threshold = threshold
        self.smooth = smooth
        self.class_weights = class_weights if class_weights is not None else 1.0

    def __call__(self, y_true, y_pred):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        true_positives = K.sum(y_true * y_pred)
        predicted_positives = K.sum(y_pred)
        precision = (true_positives + self.smooth) / (predicted_positives + self.smooth)
        return K.mean(precision)

class AverageRecall(Metric):
    def __init__(self, threshold=0.5, smooth=1e-7, class_weights=1, name="average_recall"):
        super().__init__(name=name)
        self.threshold = threshold
        self.smooth = smooth
        self.class_weights = class_weights if class_weights is not None else 1.0

    def __call__(self, y_true, y_pred):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        true_positives = K.sum(y_true * y_pred)
        possible_positives = K.sum(y_true)
        recall = (true_positives + self.smooth) / (possible_positives + self.smooth)
        return K.mean(recall)

class AverageF1Score(Metric):
    def __init__(self, threshold=0.5, smooth=1e-7, class_weights=1, name="average_f1_score"):
        super().__init__(name=name)
        self.threshold = threshold
        self.smooth = smooth
        self.class_weights = class_weights if class_weights is not None else 1.0

    def __call__(self, y_true, y_pred):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        true_positives = K.sum(y_true * y_pred)
        predicted_positives = K.sum(y_pred)
        possible_positives = K.sum(y_true)

        precision = (true_positives + self.smooth) / (predicted_positives + self.smooth)
        recall = (true_positives + self.smooth) / (possible_positives + self.smooth)
        f1_score = 2 * (precision * recall) / (precision + recall + self.smooth)
        return K.mean(f1_score)

class MeanIOU(Metric):
    def __init__(self, threshold=0.5, smooth=1e-7, name="mean_iou"):
        super().__init__(name=name)
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        intersection = K.sum(y_true * y_pred, axis=[1, 2])
        union = K.sum(y_true + y_pred, axis=[1, 2]) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return K.mean(iou)


class FocalTverskyBoundarySSIM(tf.keras.losses.Loss):
    def __init__(self, focal_tversky_weight=1, boundary_weight=1, edge_weight=0, ssim_weight=1,
                 smooth=1e-7, class_weights=None, threshold=None, alpha=0.7, beta=0.3, gamma=2.0):
        super().__init__(name="combined_loss")
        self.focal_tversky_weight = focal_tversky_weight
        self.boundary_weight = boundary_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight  # ✅ Explicitly defining SSIM weight

        self.focal_tversky_loss = self.focal_tversky_loss_fn(alpha, beta, gamma)
        self.boundary_loss = BoundaryLoss()
        self.edge_loss = MaskBoundaryLoss()
        self.ssim_loss = SSIMLoss()

    def focal_tversky_loss_fn(self, alpha=0.7, beta=0.3, gamma=2.0):
        def focal_tversky_loss(y_true, y_pred):
            tp = tf.reduce_sum(y_true * y_pred)
            fp = tf.reduce_sum((1 - y_true) * y_pred)
            fn = tf.reduce_sum(y_true * (1 - y_pred))
            tversky_index = tp / (tp + alpha * fp + beta * fn)
            focal_tversky = (1 - tversky_index) ** gamma
            return focal_tversky
        return focal_tversky_loss

    def call(self, y_true, y_pred):
        focal_tversky_loss_value = self.focal_tversky_loss(y_true, y_pred)
        boundary_loss_value = self.boundary_loss(y_true, y_pred)
        edge_loss_value = self.edge_loss(y_true, y_pred)
        ssim_loss_value = self.ssim_loss(y_true, y_pred)

        return (self.focal_tversky_weight * focal_tversky_loss_value +
                self.boundary_weight * boundary_loss_value + 
                self.edge_weight * edge_loss_value +
                self.ssim_weight * ssim_loss_value)  # ✅ Properly scaling SSIM loss


# Combined loss class
class CombinedLoss(Loss):
    def __init__(self, dice_weight=1, bce_weight=1, smooth=1e-7, class_weights=None, threshold=None):
        super().__init__(name="combined_loss")
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth, class_weights=class_weights, threshold=threshold)
        self.bce_loss = BinaryCrossEntropyLoss(class_weights=class_weights, threshold=threshold)

    def __call__(self, y_true, y_pred):
        dice_loss_value = self.dice_loss(y_true, y_pred)
        bce_loss_value = self.bce_loss(y_true, y_pred)
        return self.dice_weight * dice_loss_value + self.bce_weight * bce_loss_value

class CombinedLossBoundarySSIM(tf.keras.losses.Loss):
    def __init__(self, dice_weight=1, bce_weight=1, boundary_weight=1, edge_weight=0, ssim_weight=1,
                 smooth=1e-7, class_weights=None, threshold=None):
        super().__init__(name="combined_loss")  # Naming for tracking in logs
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight  # ✅ Explicitly defining SSIM weight

        self.dice_loss = DiceLoss(smooth=smooth, class_weights=class_weights, threshold=threshold)
        self.bce_loss = BinaryCrossEntropyLoss(class_weights=class_weights, threshold=threshold)
        self.boundary_loss = BoundaryLoss()
        self.edge_loss = MaskBoundaryLoss()
        self.ssim_loss = SSIMLoss()

    def call(self, y_true, y_pred):
        dice_loss_value = self.dice_loss(y_true, y_pred)
        bce_loss_value = self.bce_loss(y_true, y_pred)
        boundary_loss_value = self.boundary_loss(y_true, y_pred)
        edge_loss_value = self.edge_loss(y_true, y_pred)
        ssim_loss_value = self.ssim_loss(y_true, y_pred)

        return (self.dice_weight * dice_loss_value +
                self.bce_weight * bce_loss_value +
                self.boundary_weight * boundary_loss_value + 
                self.edge_weight * edge_loss_value +
                self.ssim_weight * ssim_loss_value)  # ✅ Properly scaling SSIM loss
    
class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self, max_val=1.0, name="ssim_loss"):
        super().__init__(name=name)
        self.max_val = max_val  # Maximum pixel value (1.0 for normalized images)

    def call(self, y_true, y_pred):
        return 1 - tf.image.ssim(y_true, y_pred, max_val=self.max_val)


class CombinedLossBoundary(tf.keras.losses.Loss):
    def __init__(self, dice_weight=1, bce_weight=1, boundary_weight=1, edge_weight=1,
                 smooth=1e-7, class_weights=None, threshold=None):
        super().__init__(name="combined_loss")
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.edge_weight = edge_weight


        self.dice_loss = DiceLoss(smooth=smooth, class_weights=class_weights, threshold=threshold)
        self.bce_loss = BinaryCrossEntropyLoss(class_weights=class_weights, threshold=threshold)
        self.boundary_loss = BoundaryLoss()
        self.edge_loss = MaskBoundaryLoss()

    def call(self, y_true, y_pred):
        dice_loss_value = self.dice_loss(y_true, y_pred)
        bce_loss_value = self.bce_loss(y_true, y_pred)
        boundary_loss_value = self.boundary_loss(y_true, y_pred)
        edge_loss_value = self.edge_loss(y_true, y_pred)

        return (self.dice_weight * dice_loss_value +
                self.bce_weight * bce_loss_value +
                self.boundary_weight * boundary_loss_value + 
                self.edge_weight * edge_loss_value)

class BoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, name="boundary_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Compute the boundary importance map using Sobel filters
        boundary_targets = self.sobel_edges(y_true)

        # Compute BCE loss
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)

        # Ensure BCE and boundary_targets have matching dimensions
        bce = tf.expand_dims(bce, axis=-1)

        # Compute weighted boundary loss
        boundary_loss = K.mean(bce * boundary_targets)

        return boundary_loss

    @staticmethod
    def sobel_edges(mask):
        """
        Computes edge importance map using Sobel filters.
        """
        sobel_x = tf.image.sobel_edges(mask)[..., 0]  # Horizontal gradient
        sobel_y = tf.image.sobel_edges(mask)[..., 1]  # Vertical gradient
        edges = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))  # Edge magnitude
        
        # Normalize edge map to [0,1] range
        edges = edges / (K.max(edges) + 1e-6)
        
        return edges

def compute_edge_mask(y):
    """
    Compute edge mask using Sobel filters.
    
    Args:
        y: Ground truth segmentation mask, shape (batch, H, W, 1).
    
    Returns:
        edge_mask: Binary edge mask (1 at edges, 0 elsewhere).
    """
    sobel_x = tf.image.sobel_edges(y)  # Shape: (batch, H, W, 1, 2) -> 2 for x & y gradients
    edge_magnitude = tf.sqrt(tf.reduce_sum(tf.square(sobel_x), axis=-1))  # Compute gradient magnitude
    edge_mask = tf.where(edge_magnitude > 0, 1.0, 0.0)  # Convert to binary mask (0 or 1)
    return edge_mask  # Shape: (batch, H, W, 1)

class MaskBoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, name="boundary_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Compute the boundary importance map using Sobel filters
        edge_mask = compute_edge_mask(y_true)  # Compute edge mask from ground truth

        # Compute BCE loss
        bce = tf.keras.losses.binary_crossentropy(y_true *edge_mask, y_pred *edge_mask, from_logits=False)

        # Ensure BCE and boundary_targets have matching dimensions
        bce = tf.expand_dims(bce, axis=-1)

        # Compute weighted boundary loss
        boundary_loss = K.mean(bce)

        return boundary_loss
    
def pixel_wise_contrastive_loss(teacher_pred, student_pred, margin=1.0):
    """
    Pixel-wise contrastive loss to encourage student predictions to align better with teacher predictions.
    Args:
        teacher_pred: Output from the teacher model.
        student_pred: Output from the student model.
        margin: Margin for contrastive loss.

    Returns:
        Contrastive loss value.
    """
    teacher_pred = tf.nn.sigmoid(teacher_pred)  # Ensure teacher output is in [0,1]
    student_pred = tf.nn.sigmoid(student_pred)  # Ensure student output is in [0,1]

    # Compute absolute difference
    pixel_diff = tf.abs(teacher_pred - student_pred)

    # Contrastive loss: Minimize difference for similar pixels, push apart dissimilar ones
    loss = tf.reduce_mean(tf.maximum(margin - pixel_diff, 0) ** 2)
    return loss

def nt_xent_loss(teacher_pred, student_pred, temperature=0.5):
    """
    Pixel-wise Nt-Xent loss for feature similarity in segmentation.

    Args:
        teacher_pred: Feature map from the teacher model. [batch, height, width, channels]
        student_pred: Feature map from the student model. [batch, height, width, channels]
        temperature: Scaling factor.

    Returns:
        Pixel-wise Nt-Xent loss.
    """
    # Reshape to [batch, pixels, channels]
    batch_size = tf.shape(teacher_pred)[0]
    num_pixels = tf.shape(teacher_pred)[1] * tf.shape(teacher_pred)[2]
    feature_dim = tf.shape(teacher_pred)[-1]

    teacher_pred = tf.reshape(teacher_pred, [batch_size, num_pixels, feature_dim])
    student_pred = tf.reshape(student_pred, [batch_size, num_pixels, feature_dim])

    # Normalize vectors
    teacher_pred = tf.nn.l2_normalize(teacher_pred, axis=-1)
    student_pred = tf.nn.l2_normalize(student_pred, axis=-1)

    # Compute cosine similarity for each pixel
    cosine_sim = tf.matmul(student_pred, teacher_pred, transpose_b=True)  # [batch, pixels, pixels]

    # Scale with temperature
    cosine_sim /= temperature

    # Labels: Identity matrix over pixels
    labels = tf.eye(num_pixels)

    # Compute cross-entropy loss per image, then average over batch
    nt_xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=cosine_sim))

    return nt_xent

