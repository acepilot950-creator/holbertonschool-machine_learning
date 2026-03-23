#!/usr/bin/env python3
"""Module that creates a TensorFlow inverse time decay learning rate."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Create a learning rate decay schedule using inverse time decay.

    Args:
        alpha (float): original learning rate
        decay_rate (float): decay rate
        decay_step (int): number of steps before decay

    Returns:
        tf.keras.optimizers.schedules.InverseTimeDecay
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
