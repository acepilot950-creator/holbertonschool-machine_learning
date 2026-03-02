#!/usr/bin/env python3
"""Module that calculates the inverse of a matrix."""

determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """Return the inverse of a square matrix, or None if singular."""
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    return [[adj[i][j] / det for j in range(n)] for i in range(n)]
