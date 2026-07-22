#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True
    ):
        """
        Initialize a Bayesian optimization instance.

        Args:
            f: Black-box function to optimize.
            X_init: Initial input samples of shape (t, 1).
            Y_init: Initial outputs of shape (t, 1).
            bounds: Tuple containing the minimum and maximum search bounds.
            ac_samples: Number of acquisition sample points.
            l: Length parameter of the Gaussian Process kernel.
            sigma_f: Standard deviation of the black-box function output.
            xsi: Exploration-exploitation factor.
            minimize: Whether to minimize or maximize the function.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        minimum, maximum = bounds
        self.X_s = np.linspace(
            minimum,
            maximum,
            ac_samples
        ).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize
