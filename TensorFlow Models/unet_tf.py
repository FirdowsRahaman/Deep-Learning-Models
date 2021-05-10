import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters):
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    return x


def upsample_block(x, filters, down_conn):
    upsample = layers.UpSampling2D(size=(2, 2))(x)
    up_conv1 = layers.Conv2D(filters=filters, kernel_size=(2, 2), padding='same')(upsample)
    concat = layers.concatenate([down_conn, up_conv1], axis=-1)
    up_conv2 = conv_block(concat, filters=filters)
    return up_conv2


def unet_model():
    input_layer = keras.Input(shape=(256, 256, 3))

    down1 = conv_block(input_layer, filters=64)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(down1)
    down2 = conv_block(pool1, filters=128)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(down2)
    down3 = conv_block(pool2, filters=256)
    pool3 = layers.MaxPooling2D(pool_size=(2,2))(down3)
    down4 = conv_block(pool3, filters=512)
    pool4 = layers.MaxPooling2D(pool_size=(2,2))(down4)
    down5 = conv_block(pool4, filters=1024)

    up1 = upsample_block(down5, filters=512, down_conn=down4)
    up2 = upsample_block(up1, filters=256, down_conn=down3)
    up3 = upsample_block(up2, filters=128, down_conn=down2)
    up4 = upsample_block(up3, filters=64, down_conn=down1)

    output = layers.Conv2D(1, 1, activation = 'sigmoid')(up4)
    model = keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    return model