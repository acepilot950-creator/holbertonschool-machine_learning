#!/usr/bin/env python3
"""Performs Principal Component Analysis."""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d), centered dataset.
        var: fraction of variance to maintain.

    Returns:
        W: weights matrix of shape (d, nd).
    """
    U, S, Vt = np.linalg.svd(X)

    variances = S ** 2
    total = np.sum(variances)
    cumulative = np.cumsum(variances) / total

    nd = np.searchsorted(cumulative, var) + 1

    W = Vt[:nd].T

    return W
