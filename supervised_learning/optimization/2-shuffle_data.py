#!/usr/bin/env python3
"""Module that shuffles two matrices in the same way."""

import numpy as np


def shuffle_data(X, Y):
    """Shuffle two matrices with the same permutation.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx)
        Y (numpy.ndarray): Matrix of shape (m, ny)

    Returns:
        tuple: shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])

    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return X_shuffled, Y_shuffled
