#!/usr/bin/env python3
"""Performs K-means clustering."""

import numpy as np

initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d).
        k: Number of clusters.
        iterations: Maximum number of iterations.

    Returns:
        C: numpy.ndarray of shape (k, d).
        clss: numpy.ndarray of shape (n,).
        Returns None, None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if type(k) is not int or k <= 0:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    C = initialize(X, k)

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    for _ in range(iterations):
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - C[np.newaxis, :, :],
            axis=2
        )

        clss = np.argmin(distances, axis=1)
        new_C = np.zeros_like(C)

        for cluster in range(k):
            points = X[clss == cluster]

            if points.shape[0] > 0:
                new_C[cluster] = np.mean(points, axis=0)
            else:
                new_C[cluster] = np.random.uniform(
                    minimum,
                    maximum
                )

        if np.array_equal(C, new_C):
            return C, clss

        C = new_C

    distances = np.linalg.norm(
        X[:, np.newaxis, :] - C[np.newaxis, :, :],
        axis=2
    )
    clss = np.argmin(distances, axis=1)

    return C, clss
