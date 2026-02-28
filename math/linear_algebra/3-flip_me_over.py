#!/usr/bin/env python3
"""Module that provides utilities for working with matrices."""


def matrix_shape(matrix):
    """Calculate the shape of a matrix (nested lists).

    Args:
        matrix (list): A nested list representing a matrix/tensor.

    Returns:
        list: A list of integers describing the size in each dimension.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
