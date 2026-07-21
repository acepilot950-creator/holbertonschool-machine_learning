#!/usr/bin/env python3
"""Performs a t-SNE transformation."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0,
         iterations=1000, lr=500):
    """Perform a t-SNE transformation on a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        ndims: The number of dimensions in the final representation.
        idims: The intermediate dimensionality after PCA.
        perplexity: The perplexity used to calculate P affinities.
        iterations: The number of optimization iterations.
        lr: The learning rate.

    Returns:
        A numpy.ndarray of shape (n, ndims) containing the optimized
        low-dimensional representation of X.
    """
    n = X.shape[0]

    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity)

    Y = np.random.randn(n, ndims)
    velocity = np.zeros((n, ndims))

    P *= 4

    for iteration in range(iterations):
        dY, Q = grads(Y, P)

        if iteration < 20:
            momentum = 0.5
        else:
            momentum = 0.8

        velocity = momentum * velocity - lr * dY
        Y += velocity
        Y -= np.mean(Y, axis=0)

        if (iteration + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(iteration + 1, C))

        if iteration == 99:
            P /= 4

    return Y
