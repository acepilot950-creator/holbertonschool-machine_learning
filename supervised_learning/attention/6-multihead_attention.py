#!/usr/bin/env python3
"""Multi-head attention module."""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi-head attention."""

    def __init__(self, dm, h):
        """
        Initialize the multi-head attention layer.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
        """
        super().__init__()

        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into multiple attention heads.

        Args:
            x: Tensor of shape (batch, seq_len, dm).
            batch_size: Size of the current batch.

        Returns:
            Tensor of shape (batch, h, seq_len, depth).
        """
        x = tf.reshape(
            x,
            (batch_size, -1, self.h, self.depth)
        )

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Perform multi-head attention.

        Args:
            Q: Tensor of shape (batch, seq_len_q, dk).
            K: Tensor of shape (batch, seq_len_v, dk).
            V: Tensor of shape (batch, seq_len_v, dv).
            mask: Optional attention mask.

        Returns:
            output: Tensor of shape (batch, seq_len_q, dm).
            weights: Tensor of shape
                (batch, h, seq_len_q, seq_len_v).
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        attention, weights = sdp_attention(
            Q,
            K,
            V,
            mask
        )

        attention = tf.transpose(
            attention,
            perm=[0, 2, 1, 3]
        )

        concat_attention = tf.reshape(
            attention,
            (batch_size, -1, self.dm)
        )

        output = self.linear(concat_attention)

        return output, weights
