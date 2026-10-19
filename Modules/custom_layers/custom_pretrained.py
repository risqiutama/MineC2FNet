import tensorflow as tf 
from tensorflow.keras.models import Model

import numpy as np


def average_weights(weights, num_channels):
    avg_weights = np.mean(weights, axis=-2).reshape(weights[:,:,-1:,:].shape)  
    tiled = np.tile(avg_weights, (num_channels, 1)) 
    return tiled

def average_weights_1channel(weights):
    avg_weights = np.mean(weights, axis=-2).reshape(weights[:,:,-1:,:].shape)  
    return avg_weights

def modify_weight_with_bias(model_orig, custom_model, num_channels):
    custom_model_config = custom_model.get_config()
    custom_model_layer_names = [custom_model_config["layers"][x]["name"] for x in range(len(custom_model_config["layers"]))]

    for layer in model_orig.layers:
        if layer.name in custom_model_layer_names:
            if layer.get_weights() != []:
                target_layer = custom_model.get_layer(layer.name)

                if layer.name in custom_model_layer_names[2]:
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]

                    if (num_channels >= 3):
                        weights_new_channels = np.concatenate((weights,   
                                                  average_weights(weights, num_channels - 3)),   
                                                  axis=-2)
                    elif (num_channels == 1):
                        weights_new_channels = average_weights_1channel(weights)
                    elif (num_channels == 2):
                        weights_new_channels = np.concatenate((average_weights_1channel(weights),   
                                                  average_weights(weights, num_channels - 1)),   
                                                  axis=-2)
                    else:
                        raise ValueError("channel must be equal to 1 or greater than 1")

                    target_layer.set_weights([weights_new_channels, biases])
                    target_layer.trainable = False
                else:
                    target_layer.set_weights(layer.get_weights())  
                    target_layer.trainable = False
    
    return custom_model

def modify_weight_no_bias(model_orig, custom_model, num_channels):
    custom_model_config = custom_model.get_config()
    custom_model_layer_names = [custom_model_config["layers"][x]["name"] for x in range(len(custom_model_config["layers"]))]

    for layer in model_orig.layers:
        if layer.name in custom_model_layer_names:
            if layer.get_weights() != []:
                target_layer = custom_model.get_layer(layer.name)

                if layer.name in custom_model_layer_names[2]:
                    weights = layer.get_weights()[0]

                    if (num_channels >= 3):
                        weights_new_channels = np.concatenate((weights,   
                                                  average_weights(weights, num_channels - 3)),   
                                                  axis=-2)
                    elif (num_channels == 1):
                        weights_new_channels = average_weights_1channel(weights)
                    elif (num_channels == 2):
                        weights_new_channels = np.concatenate((average_weights_1channel(weights),   
                                                  average_weights(weights, num_channels - 1)),   
                                                  axis=-2)
                    else:
                        raise ValueError("channel must be equal to 1 or greater than 1")

                    target_layer.set_weights([weights_new_channels])
                    target_layer.trainable = False
                else:
                    target_layer.set_weights(layer.get_weights())  
                    target_layer.trainable = False
    
    return custom_model

def modify_weight_no_bias2(model_orig, custom_model, num_channels):
    custom_model_config = custom_model.get_config()
    custom_model_layer_names = [custom_model_config["layers"][x]["name"] for x in range(len(custom_model_config["layers"]))]

    for layer in model_orig.layers:
        if layer.name in custom_model_layer_names:
            if layer.get_weights() != []:
                target_layer = custom_model.get_layer(layer.name)

                if layer.name in custom_model_layer_names[1]:
                    weights = layer.get_weights()[0]

                    if (num_channels >= 3):
                        weights_new_channels = np.concatenate((weights,   
                                                  average_weights(weights, num_channels - 3)),   
                                                  axis=-2)
                    elif (num_channels == 1):
                        weights_new_channels = average_weights_1channel(weights)
                    elif (num_channels == 2):
                        weights_new_channels = np.concatenate((average_weights_1channel(weights),   
                                                  average_weights(weights, num_channels - 1)),   
                                                  axis=-2)
                    else:
                        raise ValueError("channel must be equal to 1 or greater than 1")

                    target_layer.set_weights([weights_new_channels])
                    target_layer.trainable = False
                else:
                    target_layer.set_weights(layer.get_weights())  
                    target_layer.trainable = False
    
    return custom_model


def update_model(model_base_name, width, height, channels):
    model_config = model_base_name.get_config()
    model_config["layers"][0]["config"]["batch_input_shape"] = (None, height, width, channels)

    model_update = Model.from_config(model_config)

    return model_update


def get_custom_pretrained(name='ResNet50', weights='imagenet', width=128, height=128, channels=4, include_top=False, pooling=False):

    if name.lower() == "resnet50":
        if height <= 31 or width <= 31:
            raise ValueError("Height and Width shouldn't less than 32")
        base_model = tf.keras.applications.ResNet50(include_top=include_top, weights=weights, pooling=pooling)
        layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]

        model_update = update_model(base_model, width=width, height=height, channels=channels)
        model_new = modify_weight_with_bias(base_model, model_update, channels)
        layer_names = [model_new.get_layer(layer_name).output for layer_name in layer_names]

    elif name.lower() == 'densenet121':
        if height <= 31 or width <= 31:
            raise ValueError("Height and Width shouldn't less than 70")
        base_model = tf.keras.applications.DenseNet121(include_top=include_top, weights=weights, pooling=pooling)
        layer_names = ["conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"]

        model_update = update_model(base_model, width=width, height=height, channels=channels)
        model_new = modify_weight_no_bias(base_model, model_update, channels)
        layer_names = [model_new.get_layer(layer_name).output for layer_name in layer_names]

    elif name.lower() == 'mobilenetv2':
        if height <= 31 or width <= 31:
            raise ValueError("Height and Width shouldn't less than 32")
        base_model = tf.keras.applications.MobileNetV2(include_top=include_top, weights=weights, pooling=pooling)
        layer_names = ["block_1_expand_relu", "block_3_expand_relu","block_6_expand_relu", "block_13_expand_relu", "out_relu"]

        model_update = update_model(base_model, width=width, height=height, channels=channels)
        model_new = modify_weight_no_bias2(base_model, model_update, channels)
        layer_names = [model_new.get_layer(layer_name).output for layer_name in layer_names]

    elif name.lower() == 'mobilenetv3small':
        if height <= 31 or width <= 31:
            raise ValueError("Height and Width shouldn't less than 32")
        base_model = tf.keras.applications.MobileNetV3Small(include_top=include_top, weights=weights, pooling=pooling)
        layer_names = ["multiply", "re_lu_3", "multiply_1", "multiply_11", "multiply_17"]

        model_update = update_model(base_model, width=width, height=height, channels=channels)
        model_new = modify_weight_no_bias(base_model, model_update, channels)
        layer_names = [model_new.get_layer(layer_name).output for layer_name in layer_names]
    

    return model_new, layer_names





