#!/usr/bin/env python3
"""Performs K-means clustering on a dataset."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: Positive integer representing the number of clusters.
        iterations: Positive integer containing the maximum number
                    of iterations.

    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroids.
        clss: numpy.ndarray of shape (n,) containing cluster indices.
        On failure, returns None, None.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if X.shape[0] == 0:
        return None, None

    if type(k) is not int or k <= 0:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    C = np.random.uniform(
        minimum,
        maximum,
        size=(k, X.shape[1])
    )

    for _ in range(iterations):
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - C[np.newaxis, :, :],
            axis=2
        )

        clss = np.argmin(distances, axis=1)
        new_C = np.empty_like(C)

        for cluster in range(k):
            points = X[clss == cluster]

            if points.shape[0] == 0:
                new_C[cluster] = np.random.uniform(
                    minimum,
                    maximum
                )
            else:
                new_C[cluster] = np.mean(points, axis=0)

        if np.array_equal(C, new_C):
            return C, clss

        C = new_C

    distances = np.linalg.norm(
        X[:, np.newaxis, :] - C[np.newaxis, :, :],
        axis=2
    )
    clss = np.argmin(distances, axis=1)

    return C, clss
