#!/usr/bin/env python3
"""Finds the best number of GMM clusters using BIC."""

import numpy as np


expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """
    Find the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        kmin: Minimum number of clusters to test.
        kmax: Maximum number of clusters to test.
        iterations: Maximum number of EM iterations.
        tol: Non-negative tolerance for EM.
        verbose: Whether EM should print progress.

    Returns:
        best_k: Best number of clusters.
        best_result: Tuple containing pi, m, and S.
        log_likelihoods: Log likelihoods for all tested cluster sizes.
        bic_values: BIC values for all tested cluster sizes.
        Returns four None values on failure.
    """
    failure = (None, None, None, None)

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return failure

    n, d = X.shape

    if n == 0:
        return failure

    if type(kmin) is not int or kmin <= 0:
        return failure

    if kmax is None:
        kmax = n

    if type(kmax) is not int or kmax <= kmin:
        return failure

    if type(iterations) is not int or iterations <= 0:
        return failure

    if type(tol) is not float or tol < 0:
        return failure

    if type(verbose) is not bool:
        return failure

    count = kmax - kmin + 1
    log_likelihoods = np.empty(count)
    bic_values = np.empty(count)
    results = []

    for index, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, _, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return failure

        results.append((pi, m, S))
        log_likelihoods[index] = likelihood

        parameters = (
            k * d
            + k * d * (d + 1) / 2
            + k - 1
        )

        bic_values[index] = (
            parameters * np.log(n)
            - 2 * likelihood
        )

    best_index = np.argmin(bic_values)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, log_likelihoods, bic_values
