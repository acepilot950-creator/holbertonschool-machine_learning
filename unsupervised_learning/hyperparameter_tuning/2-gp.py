#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize a Gaussian process.

        Args:
            X_init: Initial input samples of shape (t, 1).
            Y_init: Initial output samples of shape (t, 1).
            l: Length parameter of the RBF kernel.
            sigma_f: Standard deviation of the function output.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculate the RBF covariance kernel matrix.

        Args:
            X1: Array of shape (m, 1).
            X2: Array of shape (n, 1).

        Returns:
            Covariance matrix of shape (m, n).
        """
        sqdist = (X1 - X2.T) ** 2

        return self.sigma_f ** 2 * np.exp(
            -sqdist / (2 * self.l ** 2)
        )

    def predict(self, X_s):
        """
        Predict the mean and variance at sample points.

        Args:
            X_s: Sample points of shape (s, 1).

        Returns:
            mu: Predicted means of shape (s,).
            sigma: Predicted variances of shape (s,).
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        covariance = K_ss - K_s.T @ K_inv @ K_s

        return mu.reshape(-1), np.diag(covariance)

    def update(self, X_new, Y_new):
        """
        Update the Gaussian process with a new sample.

        Args:
            X_new: New sample point of shape (1,).
            Y_new: New function value of shape (1,).
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
