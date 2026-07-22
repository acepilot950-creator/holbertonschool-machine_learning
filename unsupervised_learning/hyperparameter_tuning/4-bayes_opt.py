#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize Bayesian optimization.

        Args:
            f: Black-box function to optimize.
            X_init: Initial input samples of shape (t, 1).
            Y_init: Initial output samples of shape (t, 1).
            bounds: Tuple containing minimum and maximum search bounds.
            ac_samples: Number of acquisition sample points.
            l: Length parameter of the Gaussian Process kernel.
            sigma_f: Standard deviation of the function output.
            xsi: Exploration-exploitation factor.
            minimize: Whether to minimize or maximize the function.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculate the next best sample location.

        Uses the Expected Improvement acquisition function.

        Returns:
            X_next: Next best sample point of shape (1,).
            EI: Expected improvement values of shape (ac_samples,).
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            opt = np.min(self.gp.Y)
            improvement = opt - mu - self.xsi
        else:
            opt = np.max(self.gp.Y)
            improvement = mu - opt - self.xsi

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = improvement / sigma
            EI = (improvement * norm.cdf(Z) +
                  sigma * norm.pdf(Z))

        EI[sigma == 0] = 0
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
