#!/usr/bin/env python3
"""Defines a long short-term memory cell."""

import numpy as np


class LSTMCell:
    """Represents a long short-term memory cell."""

    def __init__(self, i, h, o):
        """Initialize the LSTM cell.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the output.
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: Previous hidden state of shape (m, h).
            c_prev: Previous cell state of shape (m, h).
            x_t: Input data at the current time step of shape (m, i).

        Returns:
            h_next: Next hidden state of shape (m, h).
            c_next: Next cell state of shape (m, h).
            y: Output probabilities of shape (m, o).
        """
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        forget_gate = self.sigmoid(
            np.matmul(concatenated, self.Wf) + self.bf
        )

        update_gate = self.sigmoid(
            np.matmul(concatenated, self.Wu) + self.bu
        )

        cell_candidate = np.tanh(
            np.matmul(concatenated, self.Wc) + self.bc
        )

        output_gate = self.sigmoid(
            np.matmul(concatenated, self.Wo) + self.bo
        )

        c_next = (
            forget_gate * c_prev
            + update_gate * cell_candidate
        )

        h_next = output_gate * np.tanh(c_next)

        output = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(output)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(value):
        """Calculate the sigmoid activation function."""
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
