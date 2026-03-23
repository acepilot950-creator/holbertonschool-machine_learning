#!/usr/bin/env python3
"""
Module that implements one-hot encoding
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Parameters:
    Y (numpy.ndarray): shape (m,) containing numeric class labels
    classes (int): number of classes

    Returns:
    numpy.ndarray: one-hot matrix with shape (classes, m)
    None: on failure
    """
    if not isinstance(Y, np.ndarray):
        return None

    if len(Y.shape) != 1:
        return None

    if not isinstance(classes, int) or classes <= 0:
        return None

    m = Y.shape[0]

    try:
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
