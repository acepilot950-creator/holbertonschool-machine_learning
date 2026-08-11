#!/usr/bin/env python3
"""Module for initializing a Q-table."""

import numpy as np


def q_init(env):
    """Initialize a Q-table with zeros.

    Args:
        env: FrozenLake environment instance.

    Returns:
        A numpy.ndarray containing the initialized Q-table.
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
