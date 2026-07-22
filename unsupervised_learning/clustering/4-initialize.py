#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model."""

import numpy as np


kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize parameters for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: Positive integer representing the number of clusters.

    Returns:
        pi: numpy.ndarray of shape (k,) containing cluster priors.
        m: numpy.ndarray of shape (k, d) containing cluster means.
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices.
        On failure, returns None, None, None.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if X.shape[0] == 0:
        return None, None, None

    if type(k) is not int or k <= 0:
        return None, None, None

    pi = np.full(k, 1 / k)
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))

    return pi, m, S
