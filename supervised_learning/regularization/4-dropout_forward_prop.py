#!/usr/bin/env python3
"""Forward propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        A_prev = cache["A" + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
            cache["A" + str(i)] = A
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = A * D
            A = A / keep_prob

            cache["A" + str(i)] = A
            cache["D" + str(i)] = D

    return cache
