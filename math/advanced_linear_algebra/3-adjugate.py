#!/usr/bin/env python3
"""Module that calculates the adjugate matrix of a matrix."""

cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Return the adjugate matrix of a square matrix."""
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cof = cofactor(matrix)
    return [[cof[i][j] for i in range(n)] for j in range(n)]
