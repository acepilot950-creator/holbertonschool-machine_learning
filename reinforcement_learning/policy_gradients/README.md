# Policy Gradients

This directory contains projects and exercises related to Policy Gradient methods in Reinforcement Learning.

## Description

Policy Gradient methods are reinforcement learning techniques that optimize a policy directly by adjusting its parameters in the direction that increases the expected reward.

The exercises in this directory focus on implementing basic policy functions and policy gradient algorithms using Python, NumPy, and Gymnasium.

## Requirements

* Ubuntu 20.04 LTS
* Python 3.9
* NumPy 1.25.2
* Gymnasium 0.29.1
* pycodestyle 2.11.1

All Python files:

* Start with `#!/usr/bin/env python3`
* Follow `pycodestyle`
* Include module and function documentation
* End with a new line
* Are executable

## Files

### `policy_gradient.py`

Contains functions used to implement Policy Gradient methods.

Current functions include:

* `policy(matrix, weight)`
  Computes the action probabilities for a given state matrix and weight matrix using the Softmax function.

## Usage

Example:

```python
import numpy as np

policy = __import__('policy_gradient').policy

weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01],
    [1.14374817e-04, 3.02332573e-01],
    [1.46755891e-01, 9.23385948e-02],
    [1.86260211e-01, 3.45560727e-01]
]))

state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214, 0.01636746, 0.01196594, -0.03095031]
]))

print(policy(state, weight))
```

Output:

```text
[[0.50351642 0.49648358]]
```

## Repository

* GitHub repository: `holbertonschool-machine_learning`
* Directory: `reinforcement_learning/policy_gradients`
