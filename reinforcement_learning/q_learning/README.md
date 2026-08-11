# Q-Learning

This project introduces the fundamentals of Q-learning, a model-free,
value-based reinforcement learning algorithm.

The exercises use the `FrozenLake` environment from Gymnasium to explore
how an agent can learn an optimal policy through interaction with an
environment.

## Learning Objectives

By completing this project, I aim to understand:

- What reinforcement learning is
- What an environment and an agent are
- What states and actions represent
- What a reward is
- What a policy function is
- What a value function is
- The difference between exploration and exploitation
- How the epsilon-greedy strategy works
- What Q-learning is
- How a Q-table represents action values
- How Q-values are updated during training
- How an agent can learn a policy through repeated interaction with an
  environment

## Q-Learning

Q-learning is a model-free reinforcement learning algorithm that learns
the expected value of taking an action in a particular state.

The learned values are stored in a Q-table:

```text
Q(state, action)
