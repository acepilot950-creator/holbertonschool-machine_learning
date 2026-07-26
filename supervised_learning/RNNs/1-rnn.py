#!/usr/bin/env python3
"""Performs forward propagation for a simple RNN."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: Instance of RNNCell used for forward propagation.
        X: Input data of shape (t, m, i).
        h_0: Initial hidden state of shape (m, h).

    Returns:
        H: All hidden states of shape (t + 1, m, h).
        Y: All outputs of shape (t, m, o).
    """
    t = X.shape[0]

    hidden_states = [h_0]
    outputs = []

    h_prev = h_0

    for step in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[step])
        hidden_states.append(h_next)
        outputs.append(y)
        h_prev = h_next

    H = np.array(hidden_states)
    Y = np.array(outputs)

    return H, Y
