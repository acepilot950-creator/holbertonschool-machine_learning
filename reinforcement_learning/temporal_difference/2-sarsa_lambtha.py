#!/usr/bin/env python3
"""SARSA(lambda) algorithm"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Chooses an action using the epsilon-greedy policy

    Q: numpy.ndarray containing the q-table
    state: current state
    epsilon: epsilon to use for the calculation

    Returns: the next action index
    """
    p = np.random.uniform()
    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs the SARSA(lambda) algorithm

    env: environment instance
    Q: numpy.ndarray of shape (s,a) containing the Q table
    lambtha: eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    epsilon: initial threshold for epsilon greedy
    min_epsilon: minimum value that epsilon should decay to
    epsilon_decay: decay rate for updating epsilon between episodes

    Returns: Q, the updated Q table
    """
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros(Q.shape)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])
            eligibility[state, action] += 1

            Q = Q + alpha * delta * eligibility
            eligibility = gamma * lambtha * eligibility

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * ep))

    return Q
