#!/usr/bin/env python3
"""Module that calculates the sensitivity
for each class in a confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion (numpy.ndarray): confusion matrix of shape (classes, classes)

    Returns:
        numpy.ndarray: sensitivity for each class
    """

    true_positives = np.diag(confusion)
    actual_total = np.sum(confusion, axis=1)

    sens = true_positives / actual_total

    return sens
