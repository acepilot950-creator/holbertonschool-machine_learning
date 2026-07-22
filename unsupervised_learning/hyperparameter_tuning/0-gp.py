#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize a Gaussian process.

        X_init: initial input samples of shape (t, 1)
        Y_init: initial output samples of shape (t, 1)
        l: length parameter of the RBF kernel
        sigma_f: standard deviation of the function output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculate the RBF covariance kernel matrix.

        X1 has shape (m, 1)
        X2 has shape (n, 1)

        Returns:
            Covariance matrix of shape (m, n)
        """
        sqdist = (X1 - X2.T) ** 2

        return (self.sigma_f ** 2) * np.exp(
            -sqdist / (2 * self.l ** 2)
        )
