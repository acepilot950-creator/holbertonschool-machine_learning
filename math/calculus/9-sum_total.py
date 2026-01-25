#!/usr/bin/env python3
"""Module that provides a function to calculate the sum of numbers."""


def summation_i_squared(n):
    if type(n) is not int or n < 1:
        return None
    return n * (n+1) * (2*n+1) // 6
