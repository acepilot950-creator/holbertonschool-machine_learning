#!/usr/bin/env python3
"""L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates cost with L2 regularization"""

    l2_sum = 0

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_sum += np.sum(np.square(W))

    l2_cost = cost + (lambtha / (2 * m)) * l2_sum

    return l2_cost
