import tensorflow as tf
import tensorflow.keras.backend as K
from Modules.custom_layers.blocks import ConvBn2D, FPNBlock
from Modules.custom_layers.attention import CBAM, GatedFusion, DiffusionRefinement

@tf.keras.utils.register_keras_serializable(package="CustomModels")
class FPNModel(tf.keras.models.Model):
    def __init__(self, base_model, output_layers, classes, height=None, width=None, channel=None, filters=128,
                 final_activation="sigmoid", use_batchnorm=True, backbone_trainable=False, pyramid_filters=256,
                 aggregation="concat", dropout=None, use_cbam=True, **kwargs):
        super(FPNModel, self).__init__(**kwargs)  
        self.axis = 3

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

        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)
        self.backbone.trainable = backbone_trainable

        self.fpn_block_p5 = FPNBlock(pyramid_filters)
        self.fpn_block_p4 = FPNBlock(pyramid_filters)
        self.fpn_block_p3 = FPNBlock(pyramid_filters)
        self.fpn_block_p2 = FPNBlock(pyramid_filters)

        self.cbam_p3 = CBAM() if use_cbam else tf.keras.layers.Layer()
        self.cbam_p2 = CBAM() if use_cbam else tf.keras.layers.Layer()

        self.convs = [ConvBn2D(filters, 3, padding="same", use_batchnorm=use_batchnorm) for _ in range(8)]
        self.spatialdropout = tf.keras.layers.SpatialDropout2D(dropout) if dropout else None
        self.conv_mid = ConvBn2D(filters, 3, padding="same", use_batchnorm=use_batchnorm)

        self.upsample8 = tf.keras.layers.UpSampling2D((8, 8), interpolation="nearest")
        self.upsample4 = tf.keras.layers.UpSampling2D((4, 4), interpolation="nearest")
        self.upsample2 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")

        self.concat = tf.keras.layers.Concatenate(axis=self.axis)
        self.add = tf.keras.layers.Add()
        self.final_conv = tf.keras.layers.Conv2D(classes, 3, padding="same", activation=None)
        self.activation = tf.keras.layers.Activation(final_activation)

        self.diffusion = DiffusionRefinement(filters)

    def call(self, inputs, training=None, return_features=False, **kwargs):
        features = self.backbone(inputs)
        f2, f3, f4, f5 = features[0], features[1], features[2], features[3]

        # FPN decoding
        p5 = self.fpn_block_p5(f5, f4, training=training)
        p4 = self.fpn_block_p4(p5, f3, training=training)
        p3 = self.fpn_block_p3(p4, f2, training=training)
        p2 = self.fpn_block_p2(p3, f2, training=training)

        # CBAM attention
        p3 = self.cbam_p3(p3)
        p2 = self.cbam_p2(p2)

        # Multi-scale conv predictions
        s = [self.convs[i](p, training=training) for i, p in enumerate([p5, p5, p4, p4, p3, p3, p2, p2])]

        # Use s2 (from p2) as the target resolution
        target_shape = tf.shape(s[7])[1:3]
        s2 = s[7]
        s3 = tf.image.resize(s[5], size=target_shape, method="bilinear")
        s4 = tf.image.resize(s[3], size=target_shape, method="bilinear")
        s5 = tf.image.resize(s[1], size=target_shape, method="bilinear")

        # Feature aggregation
        if self.aggregation == "concat":
            x = self.concat([s2, s3, s4, s5])
        elif self.aggregation == "sum":
            x = self.add([s2, s3, s4, s5])
        else:
            raise ValueError("aggregation must be 'concat' or 'sum'")

        if self.spatialdropout:
            x = self.spatialdropout(x, training=training)

        x = self.conv_mid(x, training=training)
        x = self.diffusion(x)
        x = self.final_conv(x)

        # Resize to match input image size
        x = tf.image.resize(x, size=tf.shape(inputs)[1:3], method='bilinear')
        pred = self.activation(x)

        if return_features:
            return pred, [f2, f3, f4, f5]
        return pred

    def forward_with_features(self, inputs, training=False):
        # Call the actual model (or modify if needed) and return both prediction and features
        pred, f2, f3, f4, f5 = self.call(inputs, training=training, return_features=True)
        return pred, f2, f3, f4, f5

    def model(self):
        x = tf.keras.Input(shape=(self.height, self.width, self.channel))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        config = super(FPNModel, self).get_config()
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
            "use_cbam": self.use_cbam
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(base_model=None, output_layers=None, **config)

    @staticmethod
    def get_custom_objects():
        return {
            "FPNModel": FPNModel,
            "ConvBn2D": ConvBn2D,
            "FPNBlock": FPNBlock,
            "CBAM": CBAM,
            "DiffusionRefinement": DiffusionRefinement
        }
