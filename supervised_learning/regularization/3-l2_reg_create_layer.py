#!/usr/bin/env python3
"""Create a layer with L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer with L2 regularization"""

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )

    return layer(prev)
