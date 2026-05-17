#!/usr/bin/env python3
"""Performs Principal Component Analysis."""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: Centered dataset of shape (n, d).
        var: Fraction of variance to maintain.

    Returns:
        Weights matrix of shape (d, nd).
    """
    U, S, Vt = np.linalg.svd(X)

    variance = S ** 2
    cumulative = np.cumsum(variance) / np.sum(variance)

    nd = np.argmax(cumulative >= var) + 1

    W = Vt[:nd].T

    return W
