#!/usr/bin/env python3
"""Module that creates a bag of words embedding matrix."""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis
            If None, all words within sentences should be used

    Returns:
        embeddings, features
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features: numpy.ndarray of shape (f,) containing the features used
    """
    tokenized = []
    for sentence in sentences:
        clean = re.sub(r"'\w*", "", sentence.lower())
        tokenized.append(re.findall(r"[a-z0-9]+", clean))

    if vocab is None:
        features = sorted(set(word for words in tokenized for word in words))
    else:
        features = list(vocab)

    index = {word: i for i, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, words in enumerate(tokenized):
        for word in words:
            if word in index:
                embeddings[i, index[word]] += 1

    return embeddings, np.array(features)
