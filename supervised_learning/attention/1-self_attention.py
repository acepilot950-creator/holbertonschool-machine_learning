#!/usr/bin/env python3
"""Self-attention module for machine translation."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates attention weights and a context vector."""

    def __init__(self, units):
        """
        Initialize the self-attention layer.

        Args:
            units: Number of hidden units in the alignment model.
        """
        super().__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Calculate the context vector and attention weights.

        Args:
            s_prev: Tensor of shape (batch, units) containing the
                previous decoder hidden state.
            hidden_states: Tensor of shape
                (batch, input_seq_len, units) containing the encoder
                hidden states.

        Returns:
            context: Tensor of shape (batch, units) containing the
                context vector.
            weights: Tensor of shape (batch, input_seq_len, 1)
                containing the attention weights.
        """
        s_prev = tf.expand_dims(s_prev, axis=1)

        score = self.V(
            tf.nn.tanh(
                self.W(s_prev) + self.U(hidden_states)
            )
        )

        weights = tf.nn.softmax(score, axis=1)

        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
