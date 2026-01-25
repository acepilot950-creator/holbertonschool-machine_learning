#!/usr/bin/env python3
"""Module that provides a function to calculate
the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """Return the integral of a polynomial.

    poly is a list of coefficients where the index represents the power of x.
    C is the integration constant.
    Returns a new list of coefficients for the integral.
    If poly or C are not valid, returns None.
    """
    if type(poly) is not list or len(poly) == 0:
        return None

    if type(C) is not int:
        return None

    for coef in poly:
        if type(coef) not in (int, float):
            return None

    integral = [C]

    for i in range(len(poly)):
        value = poly[i] / (i + 1)
        if value.is_integer():
            value = int(value)
        integral.append(value)

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
