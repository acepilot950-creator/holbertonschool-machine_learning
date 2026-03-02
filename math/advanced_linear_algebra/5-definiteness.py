#!/usr/bin/env python3
"""Module that determines the definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """Return the definiteness of a numpy.ndarray."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Must be 2D square matrix
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Must be symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"

    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    if np.all(eigenvalues < 0):
        return "Negative definite"

    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    if (np.any(eigenvalues > 0) and
            np.any(eigenvalues < 0)):
        return "Indefinite"

    return None
