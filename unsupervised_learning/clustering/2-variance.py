#!/usr/bin/env python3
"""Calculates the total intra-cluster variance."""

import numpy as np


def variance(X, C):
    """
    Calculate the total intra-cluster variance for a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        C: numpy.ndarray of shape (k, d) containing centroid means.

    Returns:
        The total intra-cluster variance, or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    distances = np.sum(
        (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
        axis=2
    )

    return np.sum(np.min(distances, axis=1))
