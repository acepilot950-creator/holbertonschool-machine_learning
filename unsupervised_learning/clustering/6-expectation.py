#!/usr/bin/env python3
"""Calculates the expectation step for a Gaussian Mixture Model."""

import numpy as np


pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step of the EM algorithm.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        pi: numpy.ndarray of shape (k,) containing cluster priors.
        m: numpy.ndarray of shape (k, d) containing cluster means.
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices.

    Returns:
        g: numpy.ndarray of shape (k, n) containing posterior
           probabilities.
        log_likelihood: Total log likelihood.
        Returns None, None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d):
        return None, None

    if S.shape != (k, d, d):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    if np.any(pi < 0):
        return None, None

    g = np.empty((k, n))

    for cluster in range(k):
        probabilities = pdf(X, m[cluster], S[cluster])

        if probabilities is None:
            return None, None

        g[cluster] = pi[cluster] * probabilities

    total = np.sum(g, axis=0)
    log_likelihood = np.sum(np.log(total))
    g = g / total

    return g, log_likelihood
