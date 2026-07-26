#!/usr/bin/env python3
"""Performs forward propagation for a deep RNN."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: List of RNNCell instances.
        X: Input data of shape (t, m, i).
        h_0: Initial hidden states of shape (l, m, h).

    Returns:
        H: All hidden states of shape (t + 1, l, m, h).
        Y: All outputs of shape (t, m, o).
    """
    t = X.shape[0]
    layers = len(rnn_cells)

    hidden_states = [h_0]
    outputs = []

    h_prev = h_0

    for step in range(t):
        current_input = X[step]
        current_hidden = []

        for layer in range(layers):
            h_next, y = rnn_cells[layer].forward(
                h_prev[layer],
                current_input
            )

            current_hidden.append(h_next)
            current_input = h_next

        h_prev = np.array(current_hidden)
        hidden_states.append(h_prev)
        outputs.append(y)

    H = np.array(hidden_states)
    Y = np.array(outputs)

    return H, Y
