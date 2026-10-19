import tensorflow as tf
from Modules.custom_layers.blocks import ConvBn2D, FPNBlock
from Modules.custom_layers.attention import CBAM, GatedFusion, DiffusionRefinement

@tf.keras.utils.register_keras_serializable(package="CustomModels")
class FPNModelDual(tf.keras.models.Model):
    def __init__(self, base_model, output_layers, classes, height=None, width=None, channel=None, filters=128,
                 final_activation="sigmoid", use_batchnorm=True, backbone_trainable=False, pyramid_filters=256,
                 aggregation="concat", dropout=None, use_cbam=True, phase_1=False, **kwargs):
        super(FPNModelDual, self).__init__(**kwargs)

        self.base_model = base_model
        self.output_layers = output_layers

        self.classes = classes
        self.height = height
        self.width = width
        self.channel = channel
        self.filters = filters
        self.final_activation = final_activation
        self.use_batchnorm = use_batchnorm
        self.backbone_trainable = backbone_trainable
        self.pyramid_filters = pyramid_filters
        self.aggregation = aggregation
        self.dropout = dropout
        self.use_cbam = use_cbam
        self.phase_1 = phase_1

        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)
        self.backbone.trainable = not phase_1

        self.coarse_backbone = tf.keras.models.clone_model(self.backbone)
        self.coarse_backbone.trainable = True  # Only train this in phase 1

        self.fpn_blocks = [FPNBlock(pyramid_filters) for _ in range(4)]
        self.gated_fusions = [GatedFusion() for _ in range(4)]

        self.cbams = [CBAM() if use_cbam else tf.keras.layers.Layer() for _ in range(2)]

        self.convs = [ConvBn2D(filters, 3, padding="same", use_batchnorm=use_batchnorm) for _ in range(8)]
        self.spatialdropout = tf.keras.layers.SpatialDropout2D(dropout) if dropout else None
        self.conv_mid = ConvBn2D(filters, 3, padding="same", use_batchnorm=use_batchnorm)

        self.upsample = [tf.keras.layers.UpSampling2D((s, s), interpolation="nearest") for s in [8, 4, 2]]

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.add = tf.keras.layers.Add()
        self.final_conv = tf.keras.layers.Conv2D(classes, 3, padding="same", activation=None)
        self.activation = tf.keras.layers.Activation(final_activation)

        self.diffusion = DiffusionRefinement(filters)

    def extract_encoder_features(self, x, backbone, training=False):
        features = backbone(x, training=training)
        return features[0], features[1], features[2], features[3]

    def call(self, inputs, training=False, return_feats=False, trigger_phase_1=None):
        # Dynamically set `phase_1` if `trigger_phase_1` is provided
        phase_1_dynamic = self.phase_1 if trigger_phase_1 is None else trigger_phase_1

        # === SET TRAINABLE FLAGS ===
        self.coarse_backbone.trainable = phase_1_dynamic
        self.backbone.trainable = not phase_1_dynamic

        if phase_1_dynamic:
            # Phase 1: freeze everything except coarse_backbone
            for layer in self.layers:
                if layer is not self.coarse_backbone:
                    layer.trainable = False
        else:
            # Phase 2: unfreeze all except coarse_backbone
            for layer in self.layers:
                layer.trainable = True
            self.coarse_backbone.trainable = False  # Reinforce freezing

        # === INPUT HANDLING ===
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            input_img, _, teacher_coarse_feats = inputs
        else:
            input_img = inputs
            teacher_coarse_feats = None

        # === FEATURE EXTRACTION ===
        if phase_1_dynamic:
            f2, f3, f4, f5 = self.extract_encoder_features(input_img, self.coarse_backbone, training=training)
            c2, c3, c4, c5 = f2, f3, f4, f5
        else:
            f2, f3, f4, f5 = self.extract_encoder_features(input_img, self.backbone, training=training)
            if teacher_coarse_feats is not None:
                c2, c3, c4, c5 = teacher_coarse_feats[:4]
            else:
                c2, c3, c4, c5 = self.extract_encoder_features(input_img, self.coarse_backbone, training=training)

        # === GATED FUSION ===
        f5 = self.gated_fusions[0](f5, c5)
        f4 = self.gated_fusions[1](f4, c4)
        f3 = self.gated_fusions[2](f3, c3)
        f2 = self.gated_fusions[3](f2, c2)

        # === FPN BLOCKS ===
        p5 = self.fpn_blocks[0](f5, f4, training=training)
        p4 = self.fpn_blocks[1](p5, f3, training=training)
        p3 = self.fpn_blocks[2](p4, f2, training=training)
        p2 = self.fpn_blocks[3](p3, f2, training=training)

        # === CBAM ATTENTION ===
        p3 = self.cbams[0](p3)
        p2 = self.cbams[1](p2)

        # === CONV AND AGGREGATION ===
        conv_inputs = [p5, p5, p4, p4, p3, p3, p2, p2]
        s = [self.convs[i](p, training=training) for i, p in enumerate(conv_inputs)]

        target_shape = tf.shape(s[7])[1:3]
        s2 = s[7]
        s3 = tf.image.resize(s[5], size=target_shape, method="bilinear")
        s4 = tf.image.resize(s[3], size=target_shape, method="bilinear")
        s5 = tf.image.resize(s[1], size=target_shape, method="bilinear")

        x = self.concat([s2, s3, s4, s5]) if self.aggregation == "concat" else self.add([s2, s3, s4, s5])

        if self.spatialdropout:
            x = self.spatialdropout(x, training=training)

        x = self.conv_mid(x, training=training)
        x = self.diffusion(x)
        x = self.final_conv(x)

        # === FINAL OUTPUT ===
        final_size = tf.shape(input_img)[1:3]
        x = tf.image.resize(x, size=final_size, method="bilinear")
        x = self.activation(x)

        # === RETURN FEATURES ===
        student_feats = {'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}
        return (x, student_feats) if return_feats else x


    def model(self):
        x = tf.keras.Input(shape=(self.height, self.width, self.channel))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        config = super(FPNModelDual, self).get_config()
        config.update({
            "classes": self.classes,
            "height": self.height,
            "width": self.width,
            "channel": self.channel,
            "filters": self.filters,
            "final_activation": self.final_activation,
            "use_batchnorm": self.use_batchnorm,
            "backbone_trainable": self.backbone_trainable,
            "pyramid_filters": self.pyramid_filters,
            "aggregation": self.aggregation,
            "dropout": self.dropout,
            "use_cbam": self.use_cbam,
            "phase_1": self.phase_1
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(base_model=None, output_layers=None, **config)

    @staticmethod
    def get_custom_objects():
        return {
            "FPNModelDual": FPNModelDual,
            "ConvBn2D": ConvBn2D,
            "FPNBlock": FPNBlock,
            "CBAM": CBAM,
            "GatedFusion": GatedFusion,
            "DiffusionRefinement": DiffusionRefinement
        }
