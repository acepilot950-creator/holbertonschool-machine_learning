#!/usr/bin/env python3
"""Performs expectation-maximization for a Gaussian Mixture Model."""

import numpy as np


initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """
    Perform expectation-maximization for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: Positive integer representing the number of clusters.
        iterations: Positive integer containing the maximum iterations.
        tol: Non-negative float containing the stopping tolerance.
        verbose: Boolean determining whether progress should be printed.

    Returns:
        pi: numpy.ndarray of shape (k,) containing the priors.
        m: numpy.ndarray of shape (k, d) containing the means.
        S: numpy.ndarray of shape (k, d, d) containing covariances.
        g: numpy.ndarray of shape (k, n) containing posteriors.
        log_likelihood: Log likelihood of the model.
        On failure, returns five None values.
    """
    failure = (None, None, None, None, None)

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return failure

    if X.shape[0] == 0:
        return failure

    if type(k) is not int or k <= 0:
        return failure

    if type(iterations) is not int or iterations <= 0:
        return failure

    if type(tol) is not float or tol < 0:
        return failure

    if type(verbose) is not bool:
        return failure

    pi, m, S = initialize(X, k)

    if pi is None:
        return failure

    g, log_likelihood = expectation(X, pi, m, S)

    if g is None:
        return failure

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(
            log_likelihood
        ))

    for iteration in range(1, iterations + 1):
        previous_likelihood = log_likelihood

        pi, m, S = maximization(X, g)

        if pi is None:
            return failure

        g, log_likelihood = expectation(X, pi, m, S)

        if g is None:
            return failure

        converged = abs(log_likelihood - previous_likelihood) <= tol

        if verbose and (iteration % 10 == 0 or
                        converged or iteration == iterations):
            print("Log Likelihood after {} iterations: {:.5f}".format(
                iteration, log_likelihood
            ))

        if converged:
            break

    return pi, m, S, g, log_likelihood
