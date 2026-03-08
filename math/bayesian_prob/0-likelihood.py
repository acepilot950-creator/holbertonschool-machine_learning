#!/usr/bin/env python3
"""Likelihood of binomial distribution"""


import numpy as np


def likelihood(x, n, P):
    """Calculates likelihood of obtaining x successes in n trials."""

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    factorial_n = np.math.factorial(n)
    factorial_x = np.math.factorial(x)
    factorial_n_x = np.math.factorial(n - x)

    comb = factorial_n / (factorial_x * factorial_n_x)

    likelihoods = comb * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
