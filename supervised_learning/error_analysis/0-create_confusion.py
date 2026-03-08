#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    """

    classes = labels.shape[1]

    confusion = np.zeros((classes, classes))

    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    for i in range(len(true_labels)):
        confusion[true_labels[i], pred_labels[i]] += 1

    return confusion
