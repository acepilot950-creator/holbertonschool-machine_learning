#!/usr/bin/env python3
"""Module that provides element-wise array addition."""


def add_arrays(arr1, arr2):
    """Add two arrays element-wise.

    Args:
        arr1 (list): First list of numbers.
        arr2 (list): Second list of numbers.

    Returns:
        list: New list with element-wise sums,
        or None if arrays have different lengths.
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
