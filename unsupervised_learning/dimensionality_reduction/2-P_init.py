#!/usr/bin/env python3
"""Initializes variables required for calculating t-SNE affinities."""

import numpy as np


def P_init(X, perplexity):
    """Initialize variables required to calculate P affinities.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        perplexity: The desired perplexity of the Gaussian distributions.

    Returns:
        D: Squared pairwise distances of shape (n, n).
        P: Zero-initialized affinity matrix of shape (n, n).
        betas: One-initialized beta values of shape (n, 1).
        H: Shannon entropy corresponding to the perplexity.
    """
    n = X.shape[0]

    squared_norms = np.sum(X ** 2, axis=1)
    D = (
        squared_norms[:, np.newaxis]
        + squared_norms[np.newaxis, :]
        - 2 * np.matmul(X, X.T)
    )

    D[D < 0] = 0
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
