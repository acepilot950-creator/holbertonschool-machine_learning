#!/usr/bin/env python3
"""TD(lambda) algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm

    env: environment instance
    V: numpy.ndarray of shape (s,) containing the value estimate
    policy: function that takes in a state and returns the next action
    lambtha: eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate

    Returns: V, the updated value estimate
    """
    for ep in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros(V.shape[0])

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1

            V = V + alpha * delta * eligibility
            eligibility = gamma * lambtha * eligibility

            state = next_state
            if terminated or truncated:
                break

    return V
