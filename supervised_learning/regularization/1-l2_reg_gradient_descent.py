#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network using
    gradient descent with L2 regularization"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W_curr = weights["W" + str(i)]

        dW = (np.matmul(dZ, A_prev.T) / m) + ((lambtha / m) * W_curr)
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            A_prev_current = cache["A" + str(i - 1)]
            dZ_prev = np.matmul(W_curr.T, dZ) * (1 - A_prev_current ** 2)

        weights["W" + str(i)] = W_curr - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db

        if i > 1:
            dZ = dZ_prev
