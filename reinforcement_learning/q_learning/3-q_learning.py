#!/usr/bin/env python3
"""Module for training an agent using Q-learning."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """Train an agent using the Q-learning algorithm.

    Args:
        env: FrozenLake environment instance.
        Q: Q-table containing action values.
        episodes: Number of episodes to train.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        min_epsilon: Minimum exploration rate.
        epsilon_decay: Rate at which epsilon decays.

    Returns:
        The updated Q-table and a list of rewards per episode.
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[new_state])
                - Q[state, action]
            )

            state = new_state
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

        epsilon = min_epsilon + (
            initial_epsilon - min_epsilon
        ) * np.exp(-epsilon_decay * episode)

    return Q, total_rewards
