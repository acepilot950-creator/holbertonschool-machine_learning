#!/usr/bin/env python3
"""DenseNet-121 module"""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture.

    Args:
        growth_rate: growth rate
        compression: compression factor

    Returns:
        The keras model
    """
    initializer = K.initializers.he_normal(seed=0)

    X = K.Input(shape=(224, 224, 3))

    batch = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(batch)
    conv = K.layers.Conv2D(
        filters=2 * growth_rate,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(act)

    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(conv)

    dense1, nb_filters = dense_block(pool, 2 * growth_rate, growth_rate, 6)
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)

    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)

    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)

    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(dense4)

    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(avg_pool)

    model = K.models.Model(inputs=X, outputs=output)

    return model
