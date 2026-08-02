#!/usr/bin/env python3
"""Transformer model for neural machine translation."""

import tensorflow as tf


def positional_encoding(maximum_position, dm):
    """Create sinusoidal positional encodings.

    Args:
        maximum_position: Maximum sequence length.
        dm: Dimensionality of the model.

    Returns:
        Tensor containing positional encodings.
    """
    positions = tf.cast(
        tf.range(maximum_position)[:, tf.newaxis],
        tf.float32
    )
    dimensions = tf.cast(
        tf.range(dm)[tf.newaxis, :],
        tf.float32
    )

    angle_rates = tf.math.pow(
        10000.0,
        -(2 * tf.floor(dimensions / 2)) / tf.cast(dm, tf.float32)
    )
    angles = positions * angle_rates

    even_mask = tf.cast(
        tf.equal(tf.math.floormod(tf.range(dm), 2), 0),
        tf.float32
    )
    odd_mask = 1.0 - even_mask

    encoding = (
        tf.sin(angles) * even_mask +
        tf.cos(angles) * odd_mask
    )

    return encoding[tf.newaxis, ...]


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate scaled dot-product attention.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        mask: Attention mask.

    Returns:
        Attention output and attention weights.
    """
    scores = tf.matmul(q, k, transpose_b=True)

    depth = tf.cast(tf.shape(k)[-1], tf.float32)
    scores /= tf.math.sqrt(depth)

    if mask is not None:
        scores += mask * -1e9

    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, v)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer."""

    def __init__(self, dm, h):
        """Initialize multi-head attention.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
        """
        super().__init__()

        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.wq = tf.keras.layers.Dense(dm)
        self.wk = tf.keras.layers.Dense(dm)
        self.wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, tensor, batch_size):
        """Split the last dimension into attention heads.

        Args:
            tensor: Tensor to split.
            batch_size: Current batch size.

        Returns:
            Tensor with separated attention heads.
        """
        tensor = tf.reshape(
            tensor,
            (batch_size, -1, self.h, self.depth)
        )

        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        """Perform multi-head attention.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Attention mask.

        Returns:
            Attention output and attention weights.
        """
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        attention, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask
        )

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(
            attention,
            (batch_size, -1, self.dm)
        )

        return self.linear(attention), weights


class EncoderBlock(tf.keras.layers.Layer):
    """Single Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize an encoder block."""
        super().__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Run the encoder block."""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        output1 = self.layernorm1(x + attention)

        feed_forward = self.dense_hidden(output1)
        feed_forward = self.dense_output(feed_forward)
        feed_forward = self.dropout2(
            feed_forward,
            training=training
        )

        return self.layernorm2(output1 + feed_forward)


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder."""

    def __init__(
        self,
        n_blocks,
        dm,
        h,
        hidden,
        input_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        """Initialize the Transformer encoder."""
        super().__init__()

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
            for _ in range(n_blocks)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Encode an input sequence."""
        sequence_length = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x


class DecoderBlock(tf.keras.layers.Layer):
    """Single Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize a decoder block."""
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(
        self,
        x,
        encoder_output,
        training,
        look_ahead_mask,
        padding_mask
    ):
        """Run the decoder block."""
        attention1, weights1 = self.mha1(
            x,
            x,
            x,
            look_ahead_mask
        )
        attention1 = self.dropout1(
            attention1,
            training=training
        )
        output1 = self.layernorm1(x + attention1)

        attention2, weights2 = self.mha2(
            output1,
            encoder_output,
            encoder_output,
            padding_mask
        )
        attention2 = self.dropout2(
            attention2,
            training=training
        )
        output2 = self.layernorm2(output1 + attention2)

        feed_forward = self.dense_hidden(output2)
        feed_forward = self.dense_output(feed_forward)
        feed_forward = self.dropout3(
            feed_forward,
            training=training
        )

        output3 = self.layernorm3(output2 + feed_forward)

        return output3, weights1, weights2


class Decoder(tf.keras.layers.Layer):
    """Transformer decoder."""

    def __init__(
        self,
        n_blocks,
        dm,
        h,
        hidden,
        target_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        """Initialize the Transformer decoder."""
        super().__init__()

        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            target_vocab,
            dm
        )
        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate)
            for _ in range(n_blocks)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(
        self,
        x,
        encoder_output,
        training,
        look_ahead_mask,
        padding_mask
    ):
        """Decode a target sequence."""
        sequence_length = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x, _, _ = block(
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Transformer network for sequence-to-sequence translation."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1
    ):
        """Initialize the Transformer network."""
        super().__init__()

        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self,
        inputs,
        target,
        training,
        encoder_mask,
        look_ahead_mask,
        decoder_mask
    ):
        """Perform a forward pass through the Transformer."""
        encoder_output = self.encoder(
            inputs,
            training,
            encoder_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            training,
            look_ahead_mask,
            decoder_mask
        )

        return self.linear(decoder_output)
