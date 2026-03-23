#!/usr/bin/env python3
"""Module that updates the learning rate using inverse time decay."""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Calculate the decayed learning rate.

    Args:
        alpha (float): original learning rate
        decay_rate (float): decay rate
        global_step (int): number of gradient descent passes
        decay_step (int): number of steps before decay

    Returns:
        float: updated learning rate
    """
    step = np.floor(global_step / decay_step)
    return alpha / (1 + decay_rate * step)
