import tensorflow as tf 
import tensorflow.keras.backend as K
from Modules.custom_layers.blocks import ConvBn2D, AtrousSeparableConvBn, AtrousSpatialPyramidPoolingV3


class DeepLabV3Plus(tf.keras.models.Model):
    def __init__(self, base_model, output_layers, classes, height=None, width=None, channel=None, filters=256,
                final_activation="softmax", backbone_trainable=False, 
                output_stride=8, dilations=[6, 12, 18],  **kwargs):
        super(DeepLabV3Plus, self).__init__()

        
        self.backbone = None
        self.classes = classes
        self.final_activation = final_activation
        self.filters = filters
        self.output_stride = output_stride
        self.dilations = dilations
        self.height = height
        self.width = width
        self.channel = channel


        if self.output_stride == 8:
            self.upsampling2d = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
            output_layers = output_layers[:3]
            self.dilations = [2 * rate for rate in dilations]
        elif self.output_stride == 16:
            self.upsampling2d = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
            output_layers = output_layers[:4]
            self.dilations = dilations
        else:
            raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))


        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        self.atrous_sep1 = AtrousSeparableConvBn(dilation=2, filters=filters, kernel_size=3)
        self.atrous_sep2 = AtrousSeparableConvBn(dilation=2, filters=filters, kernel_size=3)
        self.aspp = AtrousSpatialPyramidPoolingV3(self.dilations, filters)

        self.conv1x1bnrelu_1 = ConvBn2D(filters=filters, kernel_size=1)
        self.conv1x1bnrelu_2 = ConvBn2D(filters=64, kernel_size=1)

        self.concat = tf.keras.layers.Concatenate(axis=3)

        self.conv3x3bnrelu_1 = ConvBn2D(filters, 3)
        self.conv3x3bnrelu_2 = ConvBn2D(filters, 3)
        self.conv1x1_last = ConvBn2D(self.classes, 1, activation_model="linear")
        self.final_activation = tf.keras.layers.Activation(final_activation)
        

    def call(self, inputs, training=None, mask=None):

        if training is None:
            training = True

        x = self.backbone(inputs)[-1]
        low_level_features = self.backbone(inputs)[1]

        encoder = self.atrous_sep1(x, training)
        encoder = self.aspp(encoder, training)
        encoder = self.conv1x1bnrelu_1(encoder, training)
        encoder = self.upsampling2d(encoder)

        decoder_low_level_features = self.atrous_sep2(low_level_features, training)
        decoder_low_level_features = self.conv1x1bnrelu_2(decoder_low_level_features, training)

        decoder = self.concat([decoder_low_level_features, encoder])

        decoder = self.conv3x3bnrelu_1(decoder, training)
        decoder = self.conv3x3bnrelu_2(decoder, training)
        decoder = self.conv1x1_last(decoder, training=training)

        decoder = self.upsampling2d(decoder)
        decoder = self.final_activation(decoder)

        return decoder

    
    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, self.channel))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))