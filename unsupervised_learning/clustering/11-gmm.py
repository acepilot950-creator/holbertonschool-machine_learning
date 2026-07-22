#!/usr/bin/env python3
"""Calculates a Gaussian Mixture Model using sklearn."""

import sklearn.mixture


def gmm(X, k):
    """
    Calculate a Gaussian Mixture Model from a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: Number of Gaussian mixture components.

    Returns:
        pi: numpy.ndarray of shape (k,) containing cluster priors.
        m: numpy.ndarray of shape (k, d) containing cluster means.
        S: numpy.ndarray of shape (k, d, d) containing covariances.
        clss: numpy.ndarray of shape (n,) containing cluster indices.
        bic: Bayesian Information Criterion of the fitted model.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
