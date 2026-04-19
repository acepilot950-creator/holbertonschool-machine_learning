#!/usr/bin/env python3
"""Dense block module"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block.

    Args:
        X: output from the previous layer
        nb_filters: number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        The concatenated output of each layer within the dense block
        and the number of filters within the concatenated outputs
    """
    initializer = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        batch1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation('relu')(batch1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(act1)

        batch2 = K.layers.BatchNormalization(axis=3)(conv1)
        act2 = K.layers.Activation('relu')(batch2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=initializer
        )(act2)

        X = K.layers.Concatenate(axis=3)([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
