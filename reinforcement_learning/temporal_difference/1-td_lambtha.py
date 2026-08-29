#!/usr/bin/env python3
"""TD(lambda) algorithm."""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing value estimates
        policy: function that takes a state and returns the next action
        lambtha: eligibility trace factor
        episodes: total number of episodes
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """

    for _ in range(episodes):
        state, _ = env.reset()

        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]

            eligibility[state] += 1

            V += alpha * delta * eligibility

            eligibility *= gamma * lambtha

            state = next_state

            if terminated or truncated:
                break

    return V
