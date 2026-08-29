# Temporal Difference

This project covers reinforcement learning algorithms based on Temporal Difference (TD) methods.

Temporal Difference learning combines ideas from Monte Carlo methods and Dynamic Programming. Unlike Monte Carlo methods, TD algorithms can update value estimates before the end of an episode by bootstrapping from existing estimates.

## Learning Objectives

By the end of this project, I should be able to explain:

- What Monte Carlo methods are in reinforcement learning
- What Temporal Difference learning is
- The difference between Monte Carlo and Temporal Difference methods
- What TD(0) is
- What SARSA is
- What Q-learning is
- The difference between on-policy and off-policy learning
- What eligibility traces are
- What TD(lambda) is

## Requirements

- Python 3
- NumPy
- Gymnasium

## Files

| File | Description |
| --- | --- |
| `0-monte_carlo.py` | Implements Monte Carlo prediction for estimating the value function of a given policy |

## Monte Carlo

Monte Carlo methods learn value estimates using complete episodes.

For a state `s`, the return is calculated as:

```text
G_t = R_(t+1) + gamma * R_(t+2) + gamma^2 * R_(t+3) + ...
