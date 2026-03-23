#!/usr/bin/env python3
"""Module that sets up the RMSProp optimization algorithm in TensorFlow."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Create a TensorFlow RMSProp optimizer.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp weight.
        epsilon (float): Small number to avoid division by zero.

    Returns:
        tf.keras.optimizers.Optimizer: RMSProp optimizer.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
