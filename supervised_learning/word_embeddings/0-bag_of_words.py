#!/usr/bin/env python3
"""Creates Bag of Words embeddings for a collection of sentences."""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Create a Bag of Words embedding matrix.

    Args:
        sentences: List of sentences to analyze.
        vocab: Optional list of vocabulary words to use as features.

    Returns:
        embeddings: NumPy array containing word occurrence counts.
        features: NumPy array containing the vocabulary features.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
