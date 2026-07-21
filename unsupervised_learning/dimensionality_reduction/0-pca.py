#!/usr/bin/env python3
"""PCA module."""

import numpy as np


def pca(X, var=0.95):
    """Perform PCA while maintaining a specified fraction of variance.

    Args:
        X: A numpy.ndarray of shape (n, d) containing centered data.
        var: The fraction of the original variance to maintain.

    Returns:
        A numpy.ndarray of shape (d, nd) containing the weights matrix.
    """
    covariance = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:, indexes]

    cumulative = np.cumsum(eigenvalues)
    cumulative /= np.sum(eigenvalues)

    nd = np.searchsorted(cumulative, var) + 1

    return eigenvectors[:, :nd]
