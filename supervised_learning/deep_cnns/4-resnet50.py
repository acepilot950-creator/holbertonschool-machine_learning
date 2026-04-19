#!/usr/bin/env python3
"""ResNet-50 module"""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture.

    Returns:
        The keras model
    """
    initializer = K.initializers.he_normal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(X)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(act1)

    stage2 = projection_block(pool1, [64, 64, 256], s=1)
    stage2 = identity_block(stage2, [64, 64, 256])
    stage2 = identity_block(stage2, [64, 64, 256])

    stage3 = projection_block(stage2, [128, 128, 512], s=2)
    stage3 = identity_block(stage3, [128, 128, 512])
    stage3 = identity_block(stage3, [128, 128, 512])
    stage3 = identity_block(stage3, [128, 128, 512])

    stage4 = projection_block(stage3, [256, 256, 1024], s=2)
    stage4 = identity_block(stage4, [256, 256, 1024])
    stage4 = identity_block(stage4, [256, 256, 1024])
    stage4 = identity_block(stage4, [256, 256, 1024])
    stage4 = identity_block(stage4, [256, 256, 1024])
    stage4 = identity_block(stage4, [256, 256, 1024])

    stage5 = projection_block(stage4, [512, 512, 2048], s=2)
    stage5 = identity_block(stage5, [512, 512, 2048])
    stage5 = identity_block(stage5, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(stage5)

    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(avg_pool)

    model = K.models.Model(inputs=X, outputs=output)
    return model
