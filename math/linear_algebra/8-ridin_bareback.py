#!/usr/bin/env python3
"""Module that provides matrix multiplication."""


def mat_mul(mat1, mat2):
    """Multiply two 2D matrices.

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.

    Returns:
        list of lists: New matrix product,
        or None if matrices cannot be multiplied.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []

    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            value = 0
            for k in range(len(mat2)):
                value += mat1[i][k] * mat2[k][j]
            row.append(value)
        result.append(row)

    return result
