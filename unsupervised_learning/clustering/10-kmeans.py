#!/usr/bin/env python3
"""Performs K-means clustering using sklearn."""

import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: Number of clusters.

    Returns:
        C: numpy.ndarray of shape (k, d) containing centroid means.
        clss: numpy.ndarray of shape (n,) containing cluster indices.
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)

    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
