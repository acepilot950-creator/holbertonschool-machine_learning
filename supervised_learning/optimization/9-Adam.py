#!/usr/bin/env python3
"""Module that updates variables using the Adam optimization algorithm."""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """Update a variable using the Adam optimizer.

    Args:
        alpha (float): learning rate
        beta1 (float): weight for first moment
        beta2 (float): weight for second moment
        epsilon (float): small number to avoid division by zero
        var (numpy.ndarray): variable to update
        grad (numpy.ndarray): gradient of var
        v (numpy.ndarray): previous first moment
        s (numpy.ndarray): previous second moment
        t (int): time step for bias correction

    Returns:
        tuple: updated variable, new first moment, new second moment
    """

    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corr = v / (1 - beta1 ** t)
    s_corr = s / (1 - beta2 ** t)

    var = var - alpha * v_corr / (np.sqrt(s_corr) + epsilon)

    return var, v, s
