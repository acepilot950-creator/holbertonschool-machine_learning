#!/usr/bin/env python3
"""PCA module."""

import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a centered dataset."""
    U, S, Vt = np.linalg.svd(X)

    variance = S ** 2
    cumulative = np.cumsum(variance) / np.sum(variance)

    nd = np.argmax(cumulative >= var) + 1

    return Vt[:nd].T
