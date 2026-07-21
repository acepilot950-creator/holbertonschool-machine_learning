#!/usr/bin/env python3
"""Principal component analysis module."""

import numpy as np


def pca(X, var=0.95):
    """Perform PCA while maintaining a fraction of data variance.

    Args:
        X: A numpy.ndarray of shape (n, d) containing centered data.
        var: The fraction of variance that should be maintained.

    Returns:
        A numpy.ndarray of shape (d, nd) containing the weights matrix.
    """
    _, singular_values, vh = np.linalg.svd(X, full_matrices=False)

    cumulative = np.cumsum(singular_values)
    cumulative /= np.sum(singular_values)

    nd = np.searchsorted(cumulative, var) + 1

    return vh[:nd].T
