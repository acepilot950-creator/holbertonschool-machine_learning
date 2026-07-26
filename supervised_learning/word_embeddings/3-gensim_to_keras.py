#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a Gensim Word2Vec model to a trainable Keras layer.

    Args:
        model: A trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer initialized with the
        Word2Vec vectors.
    """
    weights = [
        model.wv[word]
        for word in model.wv.index_to_key
    ]

    weights = tf.convert_to_tensor(weights)

    embedding = tf.keras.layers.Embedding(
        input_dim=len(model.wv.index_to_key),
        output_dim=model.vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding
