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
    words = model.wv.index_to_key

    indices = [
        model.wv.key_to_index[word]
        for word in words
    ]

    weights = model.wv.vectors[indices]

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
