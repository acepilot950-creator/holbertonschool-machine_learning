#!/usr/bin/env python3
"""Module that creates a TF-IDF embedding matrix."""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    The computation follows the sklearn TfidfVectorizer convention:
    smoothed idf, idf = ln((1 + n) / (1 + df)) + 1, and L2 normalized rows.

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
    counts = np.zeros((len(sentences), len(features)))

    for i, words in enumerate(tokenized):
        for word in words:
            if word in index:
                counts[i, index[word]] += 1

    n = len(sentences)
    df = np.count_nonzero(counts, axis=0)
    idf = np.log((1 + n) / (1 + df)) + 1

    embeddings = counts * idf
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1
    embeddings = embeddings / norm

    return embeddings, np.array(features)
