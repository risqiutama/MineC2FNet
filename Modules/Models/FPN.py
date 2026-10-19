import tensorflow as tf 
import tensorflow.keras.backend as K
from Modules.custom_layers.blocks import ConvBn2D, FPNBlock


class FPNModel(tf.keras.models.Model):
    def __init__(self, base_model, output_layers, classes, height=None, width=None, channel=None, filters=128,
                final_activation="sigmoid", use_batchnorm=True, backbone_trainable=False, pyramid_filters=256,
                aggregation="concat", dropout=None, **kwargs):
        super(FPNModel, self).__init__()

        self.axis = 3 if K.image_data_format() == "channels_last" else 1

        self.backbone = None
        #self.base_model = base_model
        self.classes = classes
        self.final_activation = final_activation
        self.filters = filters
        self.pyramid_filters = pyramid_filters
        self.aggregation = aggregation
        self.dropout = dropout
        self.height = height
        self.width = width
        self.channel = channel

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        self.fpn_block_p5 = FPNBlock(pyramid_filters)
        self.fpn_block_p4 = FPNBlock(pyramid_filters)
        self.fpn_block_p3 = FPNBlock(pyramid_filters)
        self.fpn_block_p2 = FPNBlock(pyramid_filters)

        self.conv3x3bnrelu_1 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)
        self.conv3x3bnrelu_2 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)
        self.conv3x3bnrelu_3 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)
        self.conv3x3bnrelu_4 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)
        self.conv3x3bnrelu_5 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)
        self.conv3x3bnrelu_6 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)
        self.conv3x3bnrelu_7 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)
        self.conv3x3bnrelu_8 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)

        self.upsampling_s5 = tf.keras.layers.UpSampling2D((8,8), interpolation="nearest")
        self.upsampling_s4 = tf.keras.layers.UpSampling2D((4,4), interpolation="nearest")
        self.upsampling_s3 = tf.keras.layers.UpSampling2D((2,2), interpolation="nearest")

        self.add = tf.keras.layers.Add()
        self.concat = tf.keras.layers.Concatenate(axis = self.axis)

        self.spatialdropout = tf.keras.layers.SpatialDropout2D(dropout)
        self.conv3x3bnrelu_9 = ConvBn2D(filters, (3,3), padding="same", kernel_initializer="he_uniform", use_batchnorm=use_batchnorm)
        self.upsampling_last = tf.keras.layers.UpSampling2D((2,2), interpolation="bilinear")

        self.conv3x3_last = tf.keras.layers.Conv2D(self.classes, (3,3), strides=(1,1), padding="same", use_bias=True, 
                            kernel_initializer="glorot_uniform")
        self.final_activation = tf.keras.layers.Activation(final_activation)
        

    def call(self, inputs, training=None, mask=None):

        if training is None:
            training = True 
        x = self.backbone(inputs)[4]

        p5 = self.fpn_block_p5(x, self.backbone(inputs)[3], training=training)
        p4 = self.fpn_block_p4(p5, self.backbone(inputs)[2], training=training)
        p3 = self.fpn_block_p3(p4, self.backbone(inputs)[1], training=training)
        p2 = self.fpn_block_p2(p3, self.backbone(inputs)[0], training=training)

        s5 = self.conv3x3bnrelu_1(p5, training=training)
        s5 = self.conv3x3bnrelu_2(s5, training=training)
        s4 = self.conv3x3bnrelu_3(p4, training=training)
        s4 = self.conv3x3bnrelu_4(s4, training=training)
        s3 = self.conv3x3bnrelu_5(p3, training=training)
        s3 = self.conv3x3bnrelu_6(s3, training=training)
        s2 = self.conv3x3bnrelu_7(p2, training=training)
        s2 = self.conv3x3bnrelu_8(s2, training=training)

        s5 = self.upsampling_s5(s5)
        s4 = self.upsampling_s4(s4)
        s3 = self.upsampling_s3(s3)

        if self.aggregation == "sum":
            x = self.add([s2, s3, s4, s5])
        elif self.aggregation == "concat":
            x = self.concat([s2, s3, s4, s5])
        else:
            raise ValueError("Please select sum or concat")

        if self.dropout is not None:
            if self.dropout >= 1 or self.dropout < 0:
                raise ValueError("Dropout should be 0 to 1")
            else:
                x = self.spatialdropout(x, training=training)

        x = self.conv3x3bnrelu_9(x, training=training)
        x = self.upsampling_last(x)

        x = self.conv3x3_last(x)
        x = self.final_activation(x)

        return x

    
    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, self.channel))
        print(x)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

