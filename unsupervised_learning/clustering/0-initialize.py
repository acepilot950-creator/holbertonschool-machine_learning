#!/usr/bin/env python3
"""Initializes centroids for K-means clustering."""

import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: Number of clusters.

    Returns:
        numpy.ndarray of shape (k, d) containing initialized centroids.
        None if the inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if type(k) is not int or k <= 0:
        return None

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    return np.random.uniform(
        low=minimum,
        high=maximum,
        size=(k, X.shape[1])
    )
