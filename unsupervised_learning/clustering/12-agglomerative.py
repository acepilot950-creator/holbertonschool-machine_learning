#!/usr/bin/env python3
"""Performs agglomerative clustering on a dataset."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Perform agglomerative clustering using Ward linkage.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        dist: Maximum cophenetic distance for the clusters.

    Returns:
        clss: numpy.ndarray of shape (n,) containing cluster indices.
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(
        linkage,
        t=dist,
        criterion='distance'
    )

    scipy.cluster.hierarchy.dendrogram(
        linkage,
        color_threshold=dist
    )

    plt.show()

    return clss
