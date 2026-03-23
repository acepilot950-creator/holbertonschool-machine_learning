#!/usr/bin/env python3
"""Module that updates variables using the RMSProp optimization algorithm."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using RMSProp.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight
        epsilon (float): small number to avoid division by zero
        var (numpy.ndarray): variable to update
        grad (numpy.ndarray): gradient of var
        s (numpy.ndarray): previous second moment

    Returns:
        tuple: updated variable and updated second moment
    """

    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
