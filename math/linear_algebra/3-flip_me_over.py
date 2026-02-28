#!/usr/bin/env python3
"""Module that provides a function to transpose a 2D matrix."""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix.

    Args:
        matrix (list of lists): 2D matrix to transpose.

    Returns:
        list of lists: A new transposed matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
