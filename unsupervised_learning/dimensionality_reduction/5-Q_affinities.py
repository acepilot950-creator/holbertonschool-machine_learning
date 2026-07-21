#!/usr/bin/env python3
"""Calculates Q affinities for t-SNE."""

import numpy as np


def Q_affinities(Y):
    """Calculate the Q affinities for a low-dimensional dataset.

    Args:
        Y: A numpy.ndarray of shape (n, ndim) containing the
            low-dimensional representation of the data.

    Returns:
        Q: A numpy.ndarray of shape (n, n) containing Q affinities.
        num: A numpy.ndarray of shape (n, n) containing the
            unnormalized numerators of the Q affinities.
    """
    squared_norms = np.sum(Y ** 2, axis=1)

    D = (
        squared_norms[:, np.newaxis]
        + squared_norms[np.newaxis, :]
        - 2 * np.matmul(Y, Y.T)
    )

    D[D < 0] = 0

    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)

    Q = num / np.sum(num)

    return Q, num
