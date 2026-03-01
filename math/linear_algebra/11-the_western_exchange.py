#!/usr/bin/env python3
"""Module that provides a function to transpose a numpy.ndarray."""


def np_transpose(matrix):
    """Return the transpose of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Transposed array.
    """
    return matrix.T
