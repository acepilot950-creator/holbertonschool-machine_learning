#!/usr/bin/env python3
"""Module that provides a function to calculate
the derivative of a polynomial.
"""


def poly_derivative(poly):
    """Return the derivative of a polynomial.

    poly is a list of coefficients where the index represents the power of x.
    Returns a new list of coefficients for the derivative.
    If poly is not valid, returns None.
    If the derivative is zero, returns [0].
    """
    if type(poly) is not list or len(poly) == 0:
        return None

    for coef in poly:
        if type(coef) not in (int, float):
            return None

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    if not derivative or all(value == 0 for value in derivative):
        return [0]

    return derivative
