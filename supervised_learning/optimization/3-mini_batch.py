#!/usr/bin/env python3
"""Module that creates mini-batches for training."""

import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create mini-batches from shuffled data.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx).
        Y (numpy.ndarray): Labels of shape (m, ny).
        batch_size (int): Number of data points in each batch.

    Returns:
        list: List of tuples (X_batch, Y_batch).
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    mini_batches = []
    m = X.shape[0]

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
