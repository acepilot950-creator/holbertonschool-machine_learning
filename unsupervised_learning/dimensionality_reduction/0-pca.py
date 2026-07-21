#!/usr/bin/env python3
"""Module for performing principal component analysis."""

import numpy as np


def pca(X, var=0.95):
    """Calculate the PCA weights that preserve variance above var.

    Args:
        X: A numpy.ndarray of shape (n, d) containing centered data.
        var: The fraction of variance that should be maintained.

    Returns:
        A numpy.ndarray of shape (d, nd) containing the PCA weights.
    """
    _, singular_values, vh = np.linalg.svd(X, full_matrices=False)

    explained_variance = singular_values ** 2
    cumulative_variance = np.cumsum(explained_variance)
    cumulative_variance /= np.sum(explained_variance)

    nd = np.sum(cumulative_variance <= var) + 1

    return vh[:nd].T
