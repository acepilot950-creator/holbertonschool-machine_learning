#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)

            variance = 0
            for x in data:
                variance += (x - self.mean) ** 2

            variance = variance / len(data)

            self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculates the z-score of x."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of z."""
        return self.mean + z * self.stddev
