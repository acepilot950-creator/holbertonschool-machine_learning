#!/usr/bin/env python3
"""Calculates Shannon entropy and P affinities for t-SNE."""

import numpy as np


def HP(Di, beta):
    """Calculate Shannon entropy and P affinities for one data point.

    Args:
        Di: A numpy.ndarray of shape (n - 1,) containing the squared
            pairwise distances from one point to all other points.
        beta: The beta value for the Gaussian distribution.

    Returns:
        Hi: The Shannon entropy of the probability distribution.
        Pi: A numpy.ndarray of shape (n - 1,) containing P affinities.
    """
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)

    Hi = (
        np.log(sum_Pi)
        + beta * np.sum(Di * Pi) / sum_Pi
    ) / np.log(2)

    Pi = Pi / sum_Pi

    return Hi, Pi
