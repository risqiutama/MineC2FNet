import tensorflow as tf
import tensorflow.keras.backend as K


@tf.keras.utils.register_keras_serializable(package="custom_blocks")
class ConvBn2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", data_format=None,
                dilation_rate=(1, 1), groups=1, activation=None, use_bias=True, kernel_initializer="glorot_uniform",
                bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None, use_batchnorm=False, activation_model = "relu", block_name = None,
                momentum=0.99, epsilon=0.001, center=True, scale=True, **kwargs):
        super(ConvBn2D, self).__init__(**kwargs)
        self.bn_axis = 3 if K.image_data_format() == "channels_last" else 1

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = activation
        self.use_bias = not (use_batchnorm)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.block_name = block_name

        self.conv = None
        self.bn = None 
        self.activation_model = tf.keras.activations.get(activation_model)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=self.block_name + "_conv" if self.block_name is not None else None
        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.bn_axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activation_model(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
            "use_batchnorm": not self.use_bias,
            "activation_model": self.activation_model.__name__ if callable(self.activation_model) else self.activation_model,
            "block_name": self.block_name,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class FPNBlock(tf.keras.layers.Layer):
    def __init__(self, pyramid_filters, **kwargs):  
        super(FPNBlock, self).__init__(**kwargs)   
        self.pyramid_filters = pyramid_filters
        self.upsampling2d = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv1x1_x = tf.keras.layers.Conv2D(pyramid_filters, (1, 1), padding="same", kernel_initializer="he_uniform")
        self.conv1x1_skip = tf.keras.layers.Conv2D(pyramid_filters, (1, 1), padding="same", kernel_initializer="he_uniform")
        self.add = tf.keras.layers.Add()


    def call(self, x, skip, training=None):
        x = self.upsampling2d(x)
        x = self.conv1x1_x(x)
        skip = self.conv1x1_skip(skip)
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            skip = tf.image.resize(skip, (x.shape[1], x.shape[2]), method='nearest')
        return self.add([x, skip])

    def get_config(self):
        config = super().get_config()
        config.update({
            "pyramid_filters": self.pyramid_filters
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable()
class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(UpsampleBlock, self).__init__()
        self.filters = filters
        self.upsample2d = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv2x2 = tf.keras.layers.Conv2D(filters, kernel_size=(2,2), padding="same")
        self.concat = tf.keras.layers.Concatenate(axis=3)
        self.conv3x3bn_1 = ConvBn2D(filters, (3,3), activation_model='relu')
        self.conv3x3bn_2 = ConvBn2D(filters, (3,3), activation_model='relu')

    def call(self, x, skip=None, training=None):
        x = self.upsample2d(x)
        x = self.conv2x2(x, training=training)
        if skip is not None:
            x = self.concat([x, skip])
        x = self.conv3x3bn_1(x, training=training)
        x = self.conv3x3bn_2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable()
class AtrousSeparableConvBn(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=[1, 1, 1, 1], padding="SAME", data_format=None,
                 dilation=None, channel_multiplier=1, axis=-1, momentum=0.99, epsilon=0.001,
                 center=True, scale=True, trainable=True, activation_model=None, block_name=None):
        super(AtrousSeparableConvBn, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation = dilation
        self.channel_multiplier = channel_multiplier
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable
        self.block_name = block_name
        self.activation = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.dw_filter = None
        self.pw_filter = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.dw_filter = self.add_weight(
            name="dw_kernel",
            shape=[self.kernel_size, self.kernel_size, in_channels, self.channel_multiplier],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            trainable=True
        )
        self.pw_filter = self.add_weight(
            name="pw_kernel",
            shape=[1, 1, in_channels * self.channel_multiplier, self.filters],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            trainable=True
        )
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )

    def call(self, x, training=None):
        x = tf.nn.separable_conv2d(
            x, self.dw_filter, self.pw_filter,
            strides=self.strides,
            dilations=[self.dilation, self.dilation],
            padding=self.padding
        )
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation": self.dilation,
            "channel_multiplier": self.channel_multiplier,
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "trainable": self.trainable,
            "block_name": self.block_name
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable()
class AtrousSpatialPyramidPoolingV3(tf.keras.layers.Layer):
    def __init__(self, atrous_rate, filters):
        super(AtrousSpatialPyramidPoolingV3, self).__init__()
        self.atrous_rate = atrous_rate
        self.filters = filters
        self.conv_bn_relu = ConvBn2D(filters=filters, kernel_size=1)
        self.atrous_sepconv1 = AtrousSeparableConvBn(dilation=atrous_rate[0], filters=filters, kernel_size=3)
        self.atrous_sepconv2 = AtrousSeparableConvBn(dilation=atrous_rate[1], filters=filters, kernel_size=3)
        self.atrous_sepconv3 = AtrousSeparableConvBn(dilation=atrous_rate[2], filters=filters, kernel_size=3)
        self.conv_reduction = tf.keras.layers.Conv2D(filters=256, kernel_size=1, use_bias=False,
                                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    def call(self, input_tensor, training=None):
        global_avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(input_tensor)
        global_avg_pool = self.conv_bn_relu(global_avg_pool, training=training)
        global_avg_pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [input_tensor.shape[1], input_tensor.shape[2]]))(global_avg_pool)

        w = self.conv_bn_relu(input_tensor, training=training)
        x = self.atrous_sepconv1(input_tensor, training=training)
        y = self.atrous_sepconv2(input_tensor, training=training)
        z = self.atrous_sepconv3(input_tensor, training=training)

        x = tf.concat([global_avg_pool, w, x, y, z], axis=-1)
        x = self.conv_reduction(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "atrous_rate": self.atrous_rate,
            "filters": self.filters
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_custom_objects():
    return {
        "ConvBn2D": ConvBn2D,
        "FPNBlock": FPNBlock,
        "UpsampleBlock": UpsampleBlock,
        "AtrousSeparableConvBn": AtrousSeparableConvBn,
        "AtrousSpatialPyramidPoolingV3": AtrousSpatialPyramidPoolingV3
    }

