#!/usr/bin/env python3
"""Module that converts a gensim word2vec model to a keras layer."""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a trainable keras Embedding layer.

    Args:
        model: a trained gensim word2vec model

    Returns:
        The trainable keras Embedding layer
    """
    vectors = model.wv.vectors

    return tf.keras.layers.Embedding(
        input_dim=vectors.shape[0],
        output_dim=vectors.shape[1],
        weights=[vectors],
        trainable=True,
    )
