#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a Word2Vec model to a trainable Keras embedding."""
    return tf.keras.layers.Embedding(
        input_dim=len(model.wv.index_to_key),
        output_dim=model.wv.vector_size,
        embeddings_initializer=tf.keras.initializers.Constant(
            model.wv.vectors
        ),
        trainable=True
    )
