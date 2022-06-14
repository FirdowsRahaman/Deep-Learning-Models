import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend


def conv2d_bn(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', bn_axis=3):
    x = layers.Conv2D(filters=filters, 
                      kernel_size=kernel_size, 
                      strides=strides, 
                      padding=padding, 
                      use_bias=False)(
                          x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False)(x)
    x = layers.Activation('relu')(x)
    return x


def downsample_block(x):
    x_channel0 = backend.int_shape(x)[3] # 64
    x_channel1 = x_channel0 // 2         # 32
    x_channel2 = x_channel0 + x_channel1 # 96
    x_channel3 = x_channel2 // 2         # 48
    
    branch3x3dbl = conv2d_bn(x, x_channel0, (1, 1), padding='same')
    branch3x3dbl = conv2d_bn(branch3x3dbl, x_channel1, (3, 3), padding='same')
    branch3x3dbl = layers.Dropout(rate=0.3)(branch3x3dbl)
    branch3x3dbl = conv2d_bn(branch3x3dbl, x_channel1, (3, 3), padding='same')

    branch5x5 = conv2d_bn(x, x_channel2, (1, 1), padding='same')
    branch5x5 = layers.Dropout(rate=0.3)(branch5x5)
    branch5x5 = conv2d_bn(branch5x5, x_channel0, (5, 5), padding='same')

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, x_channel1, (1, 1), padding='same')

    concat = layers.concatenate([branch5x5, branch3x3dbl, branch_pool], axis=-1)
    res = conv2d_bn(x, x_channel0 * 2, (1, 1), padding='same')
    return concat + res


def upsample_block(x, filters, down_conn):
    upsample = layers.UpSampling2D(size=(2, 2))(x)
    up_conv1 = conv2d_bn(upsample, filters, kernel_size=(2, 2), padding='same')
    up_conv2 = conv2d_bn(up_conv1, filters, padding='same')
    concat = layers.concatenate([down_conn, up_conv2], axis=-1)
    up_conv3 = conv2d_bn(concat, filters=filters, padding='same')
    return up_conv3


def incepunet_2d(input_shape=(128, 128, 3), num_labels=3):
    input_layer = keras.Input(shape=input_shape)
    
    down1 = conv2d_bn(input_layer, 32, (3, 3), padding='same')
    down1 = conv2d_bn(down1, 32, (3, 3), padding='same')
    pool1 = layers.MaxPooling2D((2, 2))(down1)

    down2 = downsample_block(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(down2)
    down3 = downsample_block(pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2,2))(down3)
    down4 = downsample_block(pool3)
    pool4 = layers.MaxPooling2D(pool_size=(2,2))(down4)
    down5 = downsample_block(pool4)
    pool5 = layers.MaxPooling2D(pool_size=(2,2))(down5)
    down6 = downsample_block(pool5)

    up1 = upsample_block(down6, filters=512, down_conn=down5)
    up2 = upsample_block(up1, filters=256, down_conn=down4)
    up3 = upsample_block(up2, filters=128, down_conn=down3)
    up4 = upsample_block(up3, filters=64, down_conn=down2)
    up5 = upsample_block(up4, filters=32, down_conn=down1)

    output = layers.Conv2D(num_labels, 1, activation = 'sigmoid')(up5)
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model
