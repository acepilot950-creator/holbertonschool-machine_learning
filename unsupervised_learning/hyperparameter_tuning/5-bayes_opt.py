#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Perform Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize Bayesian optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], ac_samples
        ).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculate the next best sample location.

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
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        EI[sigma == 0] = 0
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimize the black-box function.

        Args:
            iterations: Maximum number of optimization iterations.

        Returns:
            X_opt: Optimal sampled input of shape (1,).
            Y_opt: Optimal sampled function value of shape (1,).
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(self.gp.X, X_next)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        X_opt = self.gp.X[index]
        Y_opt = self.gp.Y[index]

        return X_opt, Y_opt
