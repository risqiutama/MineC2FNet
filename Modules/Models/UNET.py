import tensorflow as tf 
import tensorflow.keras.backend as K
from Modules.custom_layers.blocks import ConvBn2D, UpsampleBlock

class Unet(tf.keras.models.Model):
    def __init__(self, base_model, output_layers, classes, height=None, width=None, channel=None, filters=128,
                final_activation="sigmoid", backbone_trainable=False, 
                up_filters=[32, 64, 128, 256, 512], include_top_conv=True,  **kwargs):
        super(Unet, self).__init__()
        
        self.backbone = None
        self.classes = classes
        self.final_activation = final_activation
        self.filters = filters
        self.up_filters = up_filters
        self.include_top_conv = include_top_conv
        self.height = height
        self.width = width
        self.channel = channel

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        self.conv3x3bnrelu_1 = ConvBn2D(filters, (3,3), activation_model='relu')
        self.conv3x3bnrelu_2 = ConvBn2D(filters, (3,3), activation_model='relu')
        
        self.upsample_1 = UpsampleBlock(up_filters[4])
        self.upsample_2 = UpsampleBlock(up_filters[3])
        self.upsample_3 = UpsampleBlock(up_filters[2])
        self.upsample_4 = UpsampleBlock(up_filters[1])
        self.upsample_5 = UpsampleBlock(up_filters[0])
        
        self.conv3x3_last = tf.keras.layers.Conv2D(self.classes, (3,3), strides=(1,1), padding="same")
        self.final_activation_layer = tf.keras.layers.Activation(final_activation) 
        
    def call(self, inputs, training=None, mask=None):

        if self.include_top_conv:
            conv1 = self.conv3x3bnrelu_1(inputs, training=training)
            conv1 = self.conv3x3bnrelu_2(conv1, training=training)
        else:
            conv1 = None 

        features = self.backbone(inputs, training=training)
        
        x = features[4]

        upsample = self.upsample_1(x, features[3], training=training)
        upsample = self.upsample_2(upsample, features[2], training=training)
        upsample = self.upsample_3(upsample, features[1], training=training)
        upsample = self.upsample_4(upsample, features[0], training=training)
        upsample = self.upsample_5(upsample, conv1, training=training)

        x = self.conv3x3_last(upsample) 
        x = self.final_activation_layer(x)

        return x
    
    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, self.channel))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))