import tensorflow as tf
from tensorflow.keras import layers
import keras

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distill_loss_tracker = keras.metrics.Mean(name="distillation_loss")
        self.contrastive_loss_tracker = keras.metrics.Mean(name="contrastive_loss")
        self.boundary_loss_tracker = keras.metrics.Mean(name="boundary_loss")  # Track boundary loss
        self.distill_ratio_tracker = keras.metrics.Mean(name="distillation_ratio")

        # Copy weights from teacher backbone to student coarse backbone
        self.copy_teacher_weights_to_coarse()

    def copy_teacher_weights_to_coarse(self):
        try:
            teacher_backbone = self.teacher.backbone
            student_coarse = self.student.coarse_backbone
        except AttributeError as e:
            print("[ERROR] Cannot access teacher.backbone or student.coarse_backbone.")
            print(e)
            return

        print("[INFO] Copying teacher backbone weights to student coarse backbone...")
        for teacher_layer, student_layer in zip(teacher_backbone.layers, student_coarse.layers):
            if isinstance(teacher_layer, type(student_layer)) and teacher_layer.name == student_layer.name:
                try:
                    student_layer.set_weights(teacher_layer.get_weights())
                    print(f"[INFO] Copied: {teacher_layer.name}")
                except Exception as ex:
                    print(f"[WARN] Skipped {teacher_layer.name} due to mismatch: {ex}")

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, boundary_loss_fn, contrastive_loss_fn=None, alpha=1, temperature=5, contrastive_weight=1, boundary_weight=1,):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.contrastive_loss_fn = contrastive_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.boundary_loss_fn = boundary_loss_fn  # Store boundary loss function
        self.boundary_weight = boundary_weight  # Store boundary weight

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False):
        images, coarse_preds, coarse_feats = x
        fine_masks = y

        teacher_pred = self.teacher(images, training=False)
        student_pred, student_feats = self.student([images, coarse_preds, coarse_feats], training=True, return_feats=True)

        # ----------------- Step 1: Per-sample loss -----------------

        def get_single_loss(loss_fn, y_true_single, y_pred_single):
            y_true_batch = tf.expand_dims(y_true_single, axis=0)
            y_pred_batch = tf.expand_dims(y_pred_single, axis=0)
            return loss_fn(y_true_batch, y_pred_batch)

        student_ce_per_sample = tf.map_fn(
            lambda elems: get_single_loss(self.student_loss_fn, elems[0], elems[1]),
            (fine_masks, student_pred),
            fn_output_signature=tf.float32
        )

        teacher_ce_per_sample = tf.map_fn(
            lambda elems: get_single_loss(self.student_loss_fn, elems[0], elems[1]),
            (fine_masks, teacher_pred),
            fn_output_signature=tf.float32
        )

        sample_mask = tf.cast(teacher_ce_per_sample < student_ce_per_sample, dtype=tf.float32)  # [batch]

        # ----------------- Step 2: Per-pixel loss -----------------

        student_ce_per_pixel = self.student_loss_fn(fine_masks, student_pred)  # [batch, H, W]
        teacher_ce_per_pixel = self.student_loss_fn(fine_masks, teacher_pred)  # [batch, H, W]

        pixel_mask = tf.cast(teacher_ce_per_pixel < student_ce_per_pixel, dtype=tf.float32)  # [batch, H, W]

        # ----------------- Step 3: Distillation loss -----------------

        teacher_softmax = tf.nn.softmax(teacher_pred / self.temperature, axis=-1)
        student_softmax = tf.nn.softmax(student_pred / self.temperature, axis=-1)

        per_pixel_distill_loss = self.distillation_loss_fn(teacher_softmax, student_softmax)  # [batch, H, W]
        per_pixel_distill_loss *= (self.temperature**2)

        # ----------------- Step 4: Combine masks -----------------

        # Expand sample_mask: [batch] → [batch, 1, 1]
        sample_mask_exp = tf.reshape(sample_mask, [-1, 1, 1])

        # Combine: sample-level AND pixel-level masks → [batch, H, W]
        combined_mask = sample_mask_exp * pixel_mask

        # ----------------- Step 5: Final distillation loss -----------------

        masked_loss = per_pixel_distill_loss * combined_mask

        # Mean over selected pixels (avoid dividing by total if all masks are 0)
        valid_pixel_count = tf.reduce_sum(combined_mask)
        distillation_loss = tf.cond(
            tf.greater(valid_pixel_count, 0),
            lambda: tf.reduce_sum(masked_loss) / valid_pixel_count,
            lambda: tf.constant(0.0)
        )

        student_loss = self.student_loss_fn(fine_masks, student_pred)

        contrastive_loss = 0.0
        if self.contrastive_loss_fn:
            contrastive_loss = self.contrastive_loss_fn(teacher_pred, student_pred)

        # Compute boundary loss for teacher
        teacher_boundary_loss = self.boundary_loss_fn(y, teacher_pred)

        total_loss = (
            self.alpha * student_loss +
            self.alpha * distillation_loss 
            #+ self.contrastive_weight * contrastive_loss
            #- self.boundary_weight * teacher_boundary_loss
        )

        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distill_loss_tracker.update_state(distillation_loss)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.compiled_metrics.update_state(fine_masks, student_pred)
        self.boundary_loss_tracker.update_state(teacher_boundary_loss)
        self.distill_ratio_tracker.update_state(tf.reduce_mean(combined_mask))

        return total_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x=x, y=y)
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        images, _, _ = x
        y_pred = self.student([images, x[1], x[2]], training=False)
        y_pred_thresh = tf.cast(y_pred > 0.5, tf.float32)
        self.compiled_metrics.update_state(y, y_pred_thresh)

        return {m.name: m.result() for m in self.metrics + self.compiled_metrics.metrics}

    def test_step(self, data):
        x, y = data
        loss = self.compute_loss(x=x, y=y)

        images, _, _ = x
        y_pred = self.student([images, x[1], x[2]], training=False)
        y_pred_thresh = tf.cast(y_pred > 0.5, tf.float32)
        self.compiled_metrics.update_state(y, y_pred_thresh)

        return {m.name: m.result() for m in self.metrics + self.compiled_metrics.metrics}

    def reset_metrics(self):
        for m in self.metrics + self.compiled_metrics.metrics:
            m.reset_states()

    def call(self, x):
        return self.student(x)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distill_loss_tracker,
            self.contrastive_loss_tracker,
            self.boundary_loss_tracker,
        ] + self.compiled_metrics.metrics
