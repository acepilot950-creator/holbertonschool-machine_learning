#!/usr/bin/env python3
"""Calculates the cost of a t-SNE transformation."""

import numpy as np


def cost(P, Q):
    """Calculate the Kullback-Leibler divergence between P and Q.

    Args:
        P: A numpy.ndarray of shape (n, n) containing P affinities.
        Q: A numpy.ndarray of shape (n, n) containing Q affinities.

    Returns:
        The cost of the t-SNE transformation.
    """
    minimum = 1e-12

    P_safe = np.maximum(P, minimum)
    Q_safe = np.maximum(Q, minimum)

    C = np.sum(P_safe * np.log(P_safe / Q_safe))

    return C
