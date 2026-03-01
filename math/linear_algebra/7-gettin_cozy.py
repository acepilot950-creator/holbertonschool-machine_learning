#!/usr/bin/env python3
"""Module that provides 2D matrix concatenation."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two 2D matrices along a given axis.

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.
        axis (int): Axis along which to concatenate (0 or 1).

    Returns:
        list of lists: New concatenated matrix,
        or None if shapes are incompatible.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:]
                for i in range(len(mat1))]

    return None
