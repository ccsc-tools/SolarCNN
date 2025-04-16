# =========================================================================
#   (c) Copyright 2025
#   All rights reserved
#   Programs written by Chunhui Xu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

import tensorflow as tf
import tensorflow_addons as tfa

def down_block(input_layer, channel):
    con2d_1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    drop_ = tf.keras.layers.Dropout(0.2)(con2d_1)
    con2d_2 = tf.keras.layers.Conv2D(filters=channel * 2, kernel_size=(3, 3), activation='relu', padding='same')(drop_)
    down_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(con2d_2)
    return down_1

def res_block(input_layer, channel, att=False):
    con2d_r11 = tf.keras.layers.Conv2D(filters=channel, kernel_size=(3, 3), padding='same')(input_layer)
    batch_r11 = tf.keras.layers.BatchNormalization()(con2d_r11)
    act_r1 = tf.keras.layers.Activation('relu')(batch_r11)
    con2d_r12 = tf.keras.layers.Conv2D(filters=channel, kernel_size=(3, 3), padding='same')(act_r1)
    batch_r12 = tf.keras.layers.BatchNormalization()(con2d_r12)
    add_r1 = tf.keras.layers.Add()([input_layer, batch_r12])
    return add_r1

def up_block(input_layer, channel):
    up_1 = tf.keras.layers.Conv2DTranspose(filters=channel, kernel_size=(2, 2), strides=(2, 2), activation=tfa.activations.mish)(input_layer)
    drop_u1 = tf.keras.layers.Dropout(0.2)(up_1)
    con2d_u10 = tf.keras.layers.Conv2D(filters=channel, kernel_size=(3, 3), activation='tanh', padding='same')(drop_u1)
    return con2d_u10

def build_model(input_shape=(256, 256, 1), channel=32):
    input_0 = tf.keras.layers.Input(shape=input_shape)
    reshape_1 = tf.keras.layers.Reshape(input_shape)(input_0)
    
    con2d_start = tf.keras.layers.Conv2D(filters=channel, kernel_size=(7, 7), activation='relu', padding='same')(reshape_1)
    down_block_out = down_block(con2d_start, channel)
    down_block_out = down_block(down_block_out, channel * 2)
    
    res_block_out = down_block_out
    for _ in range(10):
        res_block_out = res_block(res_block_out, channel * 4, True)
    
    up_block_out = up_block(res_block_out, channel * 2)
    up_block_out = up_block(up_block_out, channel)
    
    cat_out = tf.keras.layers.Concatenate(axis=-1)([up_block_out, reshape_1])
    con2d_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(cat_out)
    output_0 = tf.keras.layers.Reshape(input_shape)(con2d_output)
    
    model = tf.keras.Model(input_0, output_0)
    return model

def ssim_loss(truth, pred):
    return 1 - tf.reduce_mean(tf.image.ssim(truth, pred, max_val=2.0))

def mae_loss(truth, pred):
    return tf.reduce_mean(tf.math.abs(truth - pred))

def mix_loss(truth, pred):
    return ssim_loss(truth, pred) + mae_loss(truth, pred)

def ssim_metric(truth, pred):
    return tf.reduce_mean(tf.image.ssim(truth, pred, max_val=2.0))
