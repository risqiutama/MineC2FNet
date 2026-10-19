import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="CustomAttention")
class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.spatial_conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def build(self, input_shape):
        self.channel = input_shape[-1]
        
        # Channel attention MLP weights
        self.avg_dense_1 = layers.Dense(self.channel // self.reduction_ratio, activation='relu')
        self.avg_dense_2 = layers.Dense(self.channel)

        self.max_dense_1 = layers.Dense(self.channel // self.reduction_ratio, activation='relu')
        self.max_dense_2 = layers.Dense(self.channel)

    def call(self, x):
        # Channel Attention
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # [B, 1, 1, C]
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)  # [B, 1, 1, C]

        # Apply shared MLP (actually separate here due to TF's functional constraints)
        avg_out = self.avg_dense_2(self.avg_dense_1(avg_pool))
        max_out = self.max_dense_2(self.max_dense_1(max_pool))

        scale = tf.nn.sigmoid(avg_out + max_out)  # [B, 1, 1, C]
        x = x * scale  # Broadcasted

        # Spatial Attention
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)  # [B, H, W, 1]
        max = tf.reduce_max(x, axis=-1, keepdims=True)   # [B, H, W, 1]
        concat = tf.concat([avg, max], axis=-1)          # [B, H, W, 2]

        spatial = self.spatial_conv(concat)              # [B, H, W, 1]
        x = x * spatial                                  # Broadcasted

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"CBAM": CBAM}


class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GatedFusion, self).__init__(**kwargs)
        self.gate_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            activation='sigmoid',
            name='gate_conv'
        )

    def call(self, fine_feat, coarse_feat):
        if fine_feat.shape[1] != coarse_feat.shape[1] or fine_feat.shape[2] != coarse_feat.shape[2]:
            coarse_feat = tf.image.resize(coarse_feat, size=(fine_feat.shape[1], fine_feat.shape[2]), method='bilinear')

        concat_feats = tf.concat([fine_feat, coarse_feat], axis=-1)
        gate = self.gate_conv(concat_feats)  # shape: (B, H, W, 1)

        return gate * coarse_feat + (1.0 - gate) * fine_feat



@tf.keras.utils.register_keras_serializable(package="CustomAttention")
class DiffusionRefinement(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(DiffusionRefinement, self).__init__(**kwargs)
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.attention = CBAM()

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x + residual

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"DiffusionRefinement": DiffusionRefinement}


def get_all_custom_attention_objects():
    return {
        **CBAM.get_custom_objects(),
        **GatedFusion.get_custom_objects(),
        **DiffusionRefinement.get_custom_objects(),
    }
