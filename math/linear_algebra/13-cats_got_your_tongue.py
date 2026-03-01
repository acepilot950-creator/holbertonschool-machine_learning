#!/usr/bin/env python3
"""Module that provides matrix concatenation using NumPy."""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate two matrices along a given axis.

    Args:
        mat1 (numpy.ndarray): First matrix.
        mat2 (numpy.ndarray): Second matrix.
        axis (int): Axis along which to concatenate.

    Returns:
        numpy.ndarray: Concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
