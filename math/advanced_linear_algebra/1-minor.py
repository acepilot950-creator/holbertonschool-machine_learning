#!/usr/bin/env python3
"""Module that calculates the minor matrix of a matrix."""

from 0-determinant import determinant


def minor(matrix):
    """Return the minor matrix of a square matrix."""
    # Check if matrix is list of lists
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # Check non-empty
    if len(matrix) == 0 or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Check square
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 1x1 case
    if n == 1:
        return [[1]]

    # Compute minor matrix
    minor_matrix = []

    for i in range(n):
        row_minors = []
        for j in range(n):
            submatrix = [
                row[:j] + row[j + 1:]
                for row in (matrix[:i] + matrix[i + 1:])
            ]
            row_minors.append(determinant(submatrix))
        minor_matrix.append(row_minors)

    return minor_matrix
