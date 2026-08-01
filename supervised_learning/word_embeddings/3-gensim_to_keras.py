#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a Gensim Word2Vec model to a trainable Keras layer.

    Args:
        model: A trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer.
    """
    layer = tf.keras.layers.Embedding(
        input_dim=model.wv.vectors.shape[0],
        output_dim=model.wv.vectors.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(
            model.wv.vectors
        ),
        trainable=True
    )

    return layer
