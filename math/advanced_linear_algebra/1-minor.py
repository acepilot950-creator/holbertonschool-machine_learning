#!/usr/bin/env python3
"""Module that calculates the minor matrix of a matrix."""

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Return the minor matrix of a square matrix."""
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

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
