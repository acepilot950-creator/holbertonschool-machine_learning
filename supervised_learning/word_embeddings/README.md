# Natural Language Processing - Word Embeddings

This project covers the fundamental concepts and techniques used to
represent words and text as numerical vectors in Natural Language
Processing.

The project includes implementations of Bag of Words, TF-IDF, Word2Vec,
GloVe, FastText, and ELMo-related operations.

## Learning Objectives

By the end of this project, I should be able to explain:

- What Natural Language Processing is
- What a word embedding is
- What Bag of Words is
- What TF-IDF is
- What CBOW is
- What Skip-gram is
- What an n-gram is
- What negative sampling is
- How Word2Vec works
- How GloVe works
- How FastText works
- How ELMo works

## Natural Language Processing

Natural Language Processing, or NLP, is a field of artificial
intelligence concerned with enabling computers to process, analyze, and
understand human language.

Common NLP tasks include:

- Text classification
- Sentiment analysis
- Machine translation
- Text generation
- Named entity recognition
- Question answering

## Word Embeddings

A word embedding is a numerical vector representation of a word.

Unlike one-hot encoded vectors, word embeddings are dense,
low-dimensional vectors that can capture semantic relationships between
words. Words with similar meanings usually have similar vector
representations.

## Bag of Words

Bag of Words represents a document using the number of occurrences of
each vocabulary word.

It does not preserve grammar or word order. Each row represents a
document or sentence, and each column represents a vocabulary word.

For example, given the vocabulary:

```text
["cat", "dog", "runs"]
