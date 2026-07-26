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

    weights = [
        model.wv.get_vector(word)
        for word in words
    ]

    weights = tf.stack(weights)

    layer = tf.keras.layers.Embedding(
        input_dim=len(words),
        output_dim=model.wv.vector_size,
        embeddings_initializer=tf.keras.initializers.Constant(weights),
        trainable=True
    )

    layer.build((None,))

    return layer
