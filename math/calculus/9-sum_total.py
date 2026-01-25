#!/usr/bin/env python3
"""Module that provides a function to calculate the sum of numbers."""


def summation_i_squared(n):
    """Return the sum of squares from 1 to n.

    Computes Σ(i^2) for i = 1..n using the closed-form formula.
    Args:
        n (int): Upper bound of the summation (must be a positive integer).
    Returns:
        int: The integer value of the sum if n is valid, otherwise None.
    """
    if type(n) is not int or n < 1:
        return None
    return n * (n+1) * (2*n+1) // 6
