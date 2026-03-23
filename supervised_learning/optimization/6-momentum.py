#!/usr/bin/env python3
"""Module that sets up the momentum optimization algorithm in TensorFlow."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Create a TensorFlow optimizer for gradient descent with momentum.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Momentum weight.

    Returns:
        tf.keras.optimizers.Optimizer: SGD optimizer with momentum.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
