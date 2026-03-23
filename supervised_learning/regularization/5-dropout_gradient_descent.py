#!/usr/bin/env python3
"""Gradient descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network using gradient descent
    with Dropout regularization"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W_curr = weights["W" + str(i)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = np.matmul(W_curr.T, dZ)
            dA_prev = (dA_prev * cache["D" + str(i - 1)]) / keep_prob
            dZ_prev = dA_prev * (1 - (cache["A" + str(i - 1)] ** 2))

        weights["W" + str(i)] = W_curr - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db

        if i > 1:
            dZ = dZ_prev
