#!/usr/bin/env python3
"""Play Atari Breakout using a trained Deep Q-Learning agent."""

import gymnasium as gym
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.core import Processor


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class GymCompatibility(gym.Wrapper):
    """Make a Gymnasium environment compatible with keras-rl2."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Perform an action using the old Gym step API."""
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render()


class AtariProcessor(Processor):
    """Process Atari observations and rewards for DQN."""

    def process_observation(self, observation):
        """Resize an observation and convert it to grayscale."""
        image = Image.fromarray(observation)
        image = image.resize(INPUT_SHAPE).convert("L")
        return np.array(image, dtype="uint8")

    def process_state_batch(self, batch):
        """Normalize a batch of states."""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip rewards to the interval [-1, 1]."""
        return np.clip(reward, -1.0, 1.0)


def build_model(nb_actions):
    """Build the Deep Q-Network."""
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation("relu"))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation("relu"))

    model.add(Dense(nb_actions))
    model.add(Activation("linear"))

    return model


def main():
    """Load the trained policy and display games of Breakout."""
    env = gym.make("Breakout-v4", render_mode="human")
    env = GymCompatibility(env)

    nb_actions = env.action_space.n

    model = build_model(nb_actions)

    memory = SequentialMemory(
        limit=1000000,
        window_length=WINDOW_LENGTH
    )

    policy = GreedyQPolicy()
    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        processor=processor,
        policy=policy,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    dqn.compile(
        Adam(learning_rate=0.00025),
        metrics=["mae"]
    )

    dqn.load_weights("policy.h5")

    dqn.test(
        env,
        nb_episodes=5,
        visualize=True
    )

    env.close()


if __name__ == "__main__":
    main()
