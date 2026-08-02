#!/usr/bin/env python3
"""Transformer encoder module."""

import tensorflow as tf
positional_encoding = __import__(
    '4-positional_encoding'
).positional_encoding
EncoderBlock = __import__(
    '7-transformer_encoder_block'
).EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Creates the encoder of a transformer."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        """
        Initialize the transformer encoder.

        Args:
            N: Number of encoder blocks.
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of units in the hidden dense layer.
            input_vocab: Size of the input vocabulary.
            max_seq_len: Maximum possible sequence length.
            drop_rate: Dropout rate.
        """
        super().__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            input_vocab,
            dm
        )

        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Pass input through the transformer encoder.

        Args:
            x: Tensor of shape (batch, input_seq_len).
            training: Boolean indicating whether the model is training.
            mask: Mask applied to multi-head attention.

        Returns:
            Tensor of shape (batch, input_seq_len, dm).
        """
        input_seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x += self.positional_encoding[:input_seq_len]

        x = self.dropout(
            x,
            training=training
        )

        for block in self.blocks:
            x = block(
                x,
                training,
                mask
            )

        return x
