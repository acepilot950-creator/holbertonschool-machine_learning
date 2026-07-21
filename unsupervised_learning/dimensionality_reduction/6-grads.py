#!/usr/bin/env python3
"""Calculates the gradients for t-SNE."""

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculate the gradients of the low-dimensional representation.

    Args:
        Y: A numpy.ndarray of shape (n, ndim) containing the
            low-dimensional representation of the dataset.
        P: A numpy.ndarray of shape (n, n) containing the
            symmetric P affinities.

    Returns:
        dY: A numpy.ndarray of shape (n, ndim) containing the gradients.
        Q: A numpy.ndarray of shape (n, n) containing the Q affinities.
    """
    Q, num = Q_affinities(Y)

    PQ = (P - Q) * num

    row_sums = np.sum(PQ, axis=1)
    dY = Y * row_sums[:, np.newaxis] - np.matmul(PQ, Y)

    return dY, Q
