#!/usr/bin/env python3
"""Performs principal component analysis on a dataset."""

import numpy as np


def pca(X, var=0.95):
    """Return the PCA weights that preserve a given fraction of variance.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the centered dataset.
        var: The fraction of the original variance to preserve.

    Returns:
        A numpy.ndarray of shape (d, nd) containing the PCA weights.
    """
    _, singular_values, vh = np.linalg.svd(X, full_matrices=False)

    explained_variance = singular_values ** 2
    cumulative_variance = np.cumsum(explained_variance)
    cumulative_variance /= cumulative_variance[-1]

    nd = np.searchsorted(cumulative_variance, var) + 1

    return vh[:nd].T
