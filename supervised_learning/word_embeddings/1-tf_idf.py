#!/usr/bin/env python3
"""Creates TF-IDF embeddings for a collection of sentences."""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix.

    Args:
        sentences: List of sentences to analyze.
        vocab: Optional list of vocabulary words to use as features.

    Returns:
        embeddings: NumPy array containing TF-IDF values.
        features: NumPy array containing the vocabulary features.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
