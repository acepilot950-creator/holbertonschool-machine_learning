#!/usr/bin/env python3
"""Module that calculates the cofactor matrix of a matrix."""

minor = __import__('1-minor').minor


def cofactor(matrix):
    """Return the cofactor matrix of a square matrix."""
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 1x1 case
    if n == 1:
        return [[1]]

    minor_matrix = minor(matrix)
    cofactor_matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            sign = (-1) ** (i + j)
            row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(row)

    return cofactor_matrix
