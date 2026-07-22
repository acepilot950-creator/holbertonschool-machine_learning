#!/usr/bin/env python3
"""Calculates the PDF of a multivariate Gaussian distribution."""

import numpy as np


def pdf(X, m, S):
    """
    Calculate the PDF of a multivariate Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing data points.
        m: numpy.ndarray of shape (d,) containing the mean.
        S: numpy.ndarray of shape (d, d) containing the covariance.

    Returns:
        A numpy.ndarray of shape (n,) containing the PDF values,
        or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None

    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    d = X.shape[1]

    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        determinant = np.linalg.det(S)
        inverse = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    if determinant <= 0:
        return None

    difference = X - m
    exponent = -0.5 * np.sum(
        np.matmul(difference, inverse) * difference,
        axis=1
    )

    coefficient = 1 / np.sqrt(
        ((2 * np.pi) ** d) * determinant
    )

    P = coefficient * np.exp(exponent)

    return np.maximum(P, 1e-300)
