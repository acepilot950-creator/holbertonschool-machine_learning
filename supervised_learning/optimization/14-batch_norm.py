#!/usr/bin/env python3
"""Module that creates a batch normalization layer in TensorFlow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        n,
        use_bias=False,
        kernel_initializer=initializer
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(dense)

    return activation(batch_norm)
