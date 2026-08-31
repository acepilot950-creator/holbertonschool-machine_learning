#!/usr/bin/env python3
"""Training function for Monte-Carlo policy gradient."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Train an agent using the Monte-Carlo policy gradient method."""
    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []
        score = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, gradient = policy_gradient(state, weight)

            state, reward, terminated, truncated, _ = env.step(action)

            gradients.append(gradient)
            rewards.append(reward)
            score += reward

        for i, gradient in enumerate(gradients):
            discounted_reward = 0

            for j, reward in enumerate(rewards[i:]):
                discounted_reward += (gamma ** j) * reward

            weight += alpha * gradient * discounted_reward

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
