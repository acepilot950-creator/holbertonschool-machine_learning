#!/usr/bin/env python3
"""Determines the optimum number of clusters for K-means."""

import numpy as np


kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Test different cluster sizes and calculate their variance reduction.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        kmin: Minimum number of clusters to test, inclusive.
        kmax: Maximum number of clusters to test, inclusive.
        iterations: Maximum number of K-means iterations.

    Returns:
        results: List containing the K-means result for each value of k.
        d_vars: List containing the reduction in variance relative to kmin.
        On failure, returns None, None.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if X.shape[0] == 0:
        return None, None

    if type(kmin) is not int or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax <= 0:
        return None, None

    if kmax <= kmin:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        result = kmeans(X, k, iterations)
        results.append(result)
        variances.append(variance(X, result[0]))

    initial_variance = variances[0]
    d_vars = [initial_variance - var for var in variances]

    return results, d_vars
