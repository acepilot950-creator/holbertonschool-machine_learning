#!/usr/bin/env python3
"""Module that updates variables using momentum optimization."""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using gradient descent with momentum.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        var (numpy.ndarray): variable to be updated
        grad (numpy.ndarray): gradient of var
        v (numpy.ndarray): previous first moment

    Returns:
        tuple: updated variable and updated moment
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v

    return var, v
