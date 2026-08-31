#!/usr/bin/env python3
"""Policy gradient functions."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy using a state matrix and weight matrix."""
    z = matrix @ weight
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Compute the Monte-Carlo policy gradient."""
    probabilities = policy(state.reshape(1, -1), weight)[0]
    action = np.random.choice(len(probabilities), p=probabilities)

    one_hot = np.zeros_like(probabilities)
    one_hot[action] = 1

    gradient = np.outer(state, one_hot - probabilities)

    return action, gradient
