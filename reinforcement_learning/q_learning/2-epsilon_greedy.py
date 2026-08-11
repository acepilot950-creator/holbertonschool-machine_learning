#!/usr/bin/env python3
"""Module for epsilon-greedy action selection."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose the next action using the epsilon-greedy strategy.

    Args:
        Q: Q-table containing action values.
        state: Current state.
        epsilon: Probability of exploration.

    Returns:
        The index of the next action.
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])
