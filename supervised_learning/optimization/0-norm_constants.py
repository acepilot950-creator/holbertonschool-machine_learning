#!/usr/bin/env python3
"""Module that calculates normalization constants for a matrix."""

import numpy as np


def normalization_constants(X):
    """Calculate the mean and standard deviation of each feature in X.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx) containing the data.

    Returns:
        tuple: A tuple containing:
            - mean (numpy.ndarray): Mean of each feature.
            - std (numpy.ndarray): Standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
