#!/usr/bin/env python3
"""Module for performing semantic search on a document corpus."""

import os

import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Find the document most semantically similar to a sentence.

    Args:
        corpus_path (str): Path to the directory containing the documents.
        sentence (str): Sentence used to search the document corpus.

    Returns:
        str: The text of the document most similar to the sentence.
    """
    documents = []

    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)

        if os.path.isfile(file_path):
            with open(file_path, encoding="utf-8") as file:
                documents.append(file.read())

    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )

    embeddings = model([sentence] + documents)

    sentence_embedding = embeddings[0]
    document_embeddings = embeddings[1:]

    similarities = np.inner(
        sentence_embedding,
        document_embeddings
    )

    most_similar_index = np.argmax(similarities)

    return documents[most_similar_index]
