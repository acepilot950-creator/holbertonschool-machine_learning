#!/usr/bin/env python3
"""Calculates the maximization step for a Gaussian Mixture Model."""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step of the EM algorithm.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        g: numpy.ndarray of shape (k, n) containing posterior
           probabilities.

    Returns:
        pi: numpy.ndarray of shape (k,) containing updated priors.
        m: numpy.ndarray of shape (k, d) containing updated means.
        S: numpy.ndarray of shape (k, d, d) containing updated
           covariance matrices.
        On failure, returns None, None, None.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    if np.any(g < 0):
        return None, None, None

    totals = np.sum(g, axis=1)

    pi = totals / n
    m = np.matmul(g, X) / totals[:, np.newaxis]
    S = np.empty((k, d, d))

    for cluster in range(k):
        difference = X - m[cluster]
        weighted = difference * g[cluster, :, np.newaxis]
        S[cluster] = np.matmul(weighted.T, difference) / totals[cluster]

    return pi, m, S
