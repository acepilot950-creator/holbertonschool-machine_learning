#!/usr/bin/env python3
"""Module that normalizes a matrix using mean and standard deviation."""

import numpy as np


def normalize(X, m, s):
    """Normalize a matrix using standardization.

    Args:
        X (numpy.ndarray): Matrix of shape (d, nx) containing the data.
        m (numpy.ndarray): Mean of each feature.
        s (numpy.ndarray): Standard deviation of each feature.

    Returns:
        numpy.ndarray: The normalized matrix.
    """
    return (X - m) / s
