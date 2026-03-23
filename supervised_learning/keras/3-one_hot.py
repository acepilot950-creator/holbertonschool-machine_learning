#!/usr/bin/env python3
"""Module that converts labels to a one-hot matrix."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix

    labels: numpy array of labels
    classes: number of classes

    returns: the one-hot matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)
