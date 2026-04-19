#!/usr/bin/env python3
"""Projection block module"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block.

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12
        s: stride of the first convolution in the main path and shortcut path

    Returns:
        The activated output of the projection block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(batch1)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act1)
    batch2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(batch2)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act2)
    batch3 = K.layers.BatchNormalization(axis=3)(conv3)

    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    shortcut_batch = K.layers.BatchNormalization(axis=3)(shortcut)

    add = K.layers.Add()([batch3, shortcut_batch])
    output = K.layers.Activation('relu')(add)

    return output
