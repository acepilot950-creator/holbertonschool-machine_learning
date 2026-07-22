#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm

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
        """Initialize a Bayesian optimization instance."""
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
        mu, variance = self.gp.predict(self.X_s)
        sigma = np.sqrt(variance)

        if self.minimize:
            best = np.min(self.gp.Y)
            improvement = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            improvement = mu - best - self.xsi

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        EI[sigma == 0] = 0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
