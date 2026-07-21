#!/usr/bin/env python3
"""Module for performing principal component analysis."""

import numpy as np


def pca(X, ndim):
    """Transform a dataset using principal component analysis.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        ndim: The number of dimensions in the transformed dataset.

    Returns:
        A numpy.ndarray of shape (n, ndim) containing the transformed
        dataset.
    """
    X_centered = X - np.mean(X, axis=0)

    _, _, vh = np.linalg.svd(X_centered, full_matrices=False)

    W = vh[:ndim].T

    return np.matmul(X_centered, W)
