# Deep Q-Learning

This project implements a Deep Q-Learning agent capable of playing Atari's Breakout environment using Keras, Keras-RL2, and Gymnasium.

The agent is trained with a Deep Q-Network (DQN) that learns to estimate the expected reward for each possible action from processed game frames.

## Requirements

The project is designed to run with:

* Ubuntu 20.04 LTS
* Python 3.9
* NumPy 1.25.2
* Gymnasium 0.29.1
* TensorFlow 2.15.0
* Keras 2.15.0
* Keras-RL2 1.0.4
* Pillow 10.3.0
* h5py 3.11.0

## Installation

Install the required dependencies:

```bash
pip install --user keras-rl2==1.0.4
pip install --user gymnasium[atari]==0.29.1
pip install --user tensorflow==2.15.0
pip install --user keras==2.15.0
pip install --user numpy==1.25.2
pip install --user Pillow==10.3.0
pip install --user h5py==3.11.0
pip install autorom[accept-rom-license]
```

## Project Files

### `train.py`

Trains a Deep Q-Learning agent to play Atari Breakout.

The script uses:

* `DQNAgent` for Deep Q-Learning
* `SequentialMemory` for experience replay
* `EpsGreedyQPolicy` for exploration during training
* A convolutional neural network to process game frames

The Atari observations are resized to `84 x 84` pixels and converted to grayscale before being passed to the neural network.

The agent uses the four most recent frames as its state in order to detect movement and direction.

After training, the policy network weights are saved as:

```text
policy.h5
```

### `play.py`

Loads the trained policy stored in `policy.h5` and uses it to play Breakout.

During evaluation, the agent uses `GreedyQPolicy`, meaning that it always selects the action with the highest predicted Q-value.

The environment is rendered so that the trained agent can be observed playing the game.

## Deep Q-Learning

Q-Learning attempts to learn an action-value function:

```text
Q(state, action)
```

which estimates the expected future reward obtained by taking an action in a particular state.

Deep Q-Learning replaces the traditional Q-table with a neural network.

For Breakout, the network receives processed game frames and produces one Q-value for each available action.

The agent then selects an action according to its policy.

## Experience Replay

The project uses `SequentialMemory` to store previous experiences.

Each experience contains information about:

```text
state
action
reward
next state
terminal state
```

Random samples from this memory are used during training.

Experience replay helps reduce correlations between consecutive observations and improves training stability.

## Exploration

During training, the agent uses `EpsGreedyQPolicy`.

Most of the time, the agent selects the action with the highest predicted Q-value. With a small probability, it selects a random action instead.

This allows the agent to balance:

```text
exploration
vs.
exploitation
```

During evaluation in `play.py`, `GreedyQPolicy` is used so that the agent always selects the action considered best by the trained network.

## Gymnasium Compatibility

Keras-RL2 was originally designed for older versions of Gym.

Gymnasium uses a different API for `reset()` and `step()`.

The project therefore uses a compatibility wrapper that converts Gymnasium's API into the format expected by Keras-RL2.

Gymnasium returns:

```python
observation, info = env.reset()
```

and:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

The wrapper converts this into the older API expected by Keras-RL2.

## Usage

Train the agent:

```bash
./train.py
```

or:

```bash
python3 train.py
```

After training, the file `policy.h5` will contain the trained policy network weights.

To watch the trained agent play Breakout:

```bash
./play.py
```

or:

```bash
python3 play.py
```

## Repository

GitHub repository:

```text
holbertonschool-machine_learning
```

Directory:

```text
reinforcement_learning/deep_q_learning
```
