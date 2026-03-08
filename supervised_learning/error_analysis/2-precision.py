#!/usr/bin/env python3
"""Module that calculates the precision
for each class in a confusion matrix
"""

import numpy as np


def precision(confusion):
    """Calculates the precision for each class

    Args:
        confusion (numpy.ndarray): confusion matrix of
        shape (classes, classes)

    Returns:
        numpy.ndarray: precision of each class
    """

    true_positives = np.diag(confusion)
    predicted_total = np.sum(confusion, axis=0)

    prec = true_positives / predicted_total

    return prec
