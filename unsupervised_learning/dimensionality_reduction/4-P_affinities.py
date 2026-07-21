#!/usr/bin/env python3
"""Calculates symmetric P affinities for t-SNE."""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculate the symmetric P affinities of a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        tol: The maximum allowed difference between the calculated
            entropy and the target entropy.
        perplexity: The desired perplexity of each Gaussian
            distribution.

    Returns:
        A numpy.ndarray of shape (n, n) containing the symmetric
        P affinities.
    """
    D, P, betas, H = P_init(X, perplexity)
    n = X.shape[0]

    for i in range(n):
        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        beta_min = None
        beta_max = None

        Hi, Pi = HP(Di, betas[i])

        while abs(Hi - H) > tol:
            if Hi > H:
                beta_min = betas[i].copy()

                if beta_max is None:
                    betas[i] *= 2
                else:
                    betas[i] = (betas[i] + beta_max) / 2
            else:
                beta_max = betas[i].copy()

                if beta_min is None:
                    betas[i] /= 2
                else:
                    betas[i] = (betas[i] + beta_min) / 2

            Hi, Pi = HP(Di, betas[i])

        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    P = (P + P.T) / (2 * n)

    return P
