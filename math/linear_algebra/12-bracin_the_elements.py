#!/usr/bin/env python3
"""Module that provides element-wise operations on numpy arrays."""


def np_elementwise(mat1, mat2):
    """Perform element-wise addition, subtraction,
    multiplication, and division.

    Args:
        mat1 (numpy.ndarray): First array.
        mat2 (numpy.ndarray or scalar): Second array or scalar.

    Returns:
        tuple: (sum, difference, product, quotient)
    """
    return (mat1 + mat2,
            mat1 - mat2,
            mat1 * mat2,
            mat1 / mat2)
