#!/usr/bin/env python3
"""Poisson distribution"""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson distribution."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the PMF value for k."""
        k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return (e ** (-self.lambtha) * self.lambtha ** k) / factorial

    def cdf(self, k):
        """Calculates the CDF value for k."""
        k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285
        cdf = 0

        for i in range(k + 1):
            factorial = 1
            for j in range(1, i + 1):
                factorial *= j

            cdf += (e ** (-self.lambtha) * self.lambtha ** i) / factorial

        return cdf
