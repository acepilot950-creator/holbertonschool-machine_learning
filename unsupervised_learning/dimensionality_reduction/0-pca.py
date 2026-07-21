#!/usr/bin/env python3
"""Principal component analysis module."""

import numpy as np


def pca(X, var=0.95):
    """Calculate PCA weights that maintain more than var of the variance.

    Args:
        X: A numpy.ndarray of shape (n, d) containing centered data.
        var: The fraction of the original variance to maintain.

    Returns:
        A numpy.ndarray of shape (d, nd) containing the weights matrix.
    """
    _, singular_values, vh = np.linalg.svd(X, full_matrices=False)

    variances = singular_values ** 2
    cumulative_variance = np.cumsum(variances) / np.sum(variances)

    nd = np.searchsorted(cumulative_variance, var, side='right') + 1

    return vh[:nd].T
