#!/usr/bin/env python3
"""Module that creates a batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer for a neural network.

    Args:
        prev: Activated output of the previous layer.
        n (int): Number of nodes in the layer.
        activation: Activation function to use.

    Returns:
        Tensor: Activated output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    Z = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )

    return activation(Z_norm)
