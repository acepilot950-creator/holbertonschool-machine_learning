#!/usr/bin/env python3
"""Module that contains the function to create a confusion matrix
for classification models.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels (numpy.ndarray): one-hot array of correct labels
                                with shape (m, classes)
        logits (numpy.ndarray): one-hot array of predicted labels
                                with shape (m, classes)

    Returns:
        numpy.ndarray: confusion matrix of shape (classes, classes)
        where rows represent correct labels and columns represent
        predicted labels.
    """

    classes = labels.shape[1]

    confusion = np.zeros((classes, classes))

    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    for i in range(len(true_labels)):
        confusion[true_labels[i], pred_labels[i]] += 1

    return confusion
