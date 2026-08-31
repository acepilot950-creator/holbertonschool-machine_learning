#!/usr/bin/env python3
"""Policy gradient functions."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy using a state matrix and weight matrix."""
    z = matrix @ weight
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)
