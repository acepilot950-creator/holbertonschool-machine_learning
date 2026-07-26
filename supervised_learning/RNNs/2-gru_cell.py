#!/usr/bin/env python3
"""Defines a gated recurrent unit cell."""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell."""

    def __init__(self, i, h, o):
        """Initialize the GRU cell.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the output.
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: Previous hidden state of shape (m, h).
            x_t: Input data at the current time step of shape (m, i).

        Returns:
            h_next: Next hidden state of shape (m, h).
            y: Output probabilities of shape (m, o).
        """
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(
            np.matmul(concatenated, self.Wz) + self.bz
        )

        r = self.sigmoid(
            np.matmul(concatenated, self.Wr) + self.br
        )

        candidate_input = np.concatenate(
            (r * h_prev, x_t),
            axis=1
        )

        h_inter = np.tanh(
            np.matmul(candidate_input, self.Wh) + self.bh
        )

        h_next = (1 - z) * h_prev + z * h_inter

        output = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(output)

        return h_next, y

    @staticmethod
    def sigmoid(value):
        """Calculate the sigmoid activation."""
        return 1 / (1 + np.exp(-value))

    @staticmethod
    def softmax(value):
        """Calculate a numerically stable softmax activation."""
        shifted = value - np.max(value, axis=1, keepdims=True)
        exponentials = np.exp(shifted)

        return exponentials / np.sum(
            exponentials,
            axis=1,
            keepdims=True
        )
