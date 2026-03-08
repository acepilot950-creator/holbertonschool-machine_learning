#!/usr/bin/env python3
"""Module that calculates the specificity
for each class in a confusion matrix
"""

import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class

    Args:
        confusion (numpy.ndarray): confusion matrix
        of shape (classes, classes)

    Returns:
        numpy.ndarray: specificity of each class
    """

    TP = np.diag(confusion)

    FP = np.sum(confusion, axis=0) - TP

    FN = np.sum(confusion, axis=1) - TP

    total = np.sum(confusion)

    TN = total - (TP + FP + FN)

    spec = TN / (TN + FP)

    return spec
