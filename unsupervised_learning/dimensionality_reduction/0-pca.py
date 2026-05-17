#!/usr/bin/env python3
"""PCA module."""

import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a centered dataset."""
    U, S, Vt = np.linalg.svd(X)

    s = np.cumsum(S ** 2)
    total = np.sum(S ** 2)

    nd = np.sum(s / total < var) + 1

    W = Vt[:nd].T

    return W
