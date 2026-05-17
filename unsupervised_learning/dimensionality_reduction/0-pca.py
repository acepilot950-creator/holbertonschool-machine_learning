#!/usr/bin/env python3
"""PCA module."""

import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset."""
    U, S, Vt = np.linalg.svd(X)

    cumulative = np.cumsum(S) / np.sum(S)
    nd = np.where(cumulative >= var)[0][0] + 1

    return Vt[:nd].T
