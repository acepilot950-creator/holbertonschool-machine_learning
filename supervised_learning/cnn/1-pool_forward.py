#!/usr/bin/env python3
"""Pooling forward propagation for a neural network."""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer.

    Args:
        A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev)
            containing output of previous layer
        kernel_shape (tuple): (kh, kw) size of pooling kernel
        stride (tuple): (sh, sw) strides
        mode (str): 'max' or 'avg'

    Returns:
        numpy.ndarray: pooled output
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):

            h_start = i * sh
            h_end = h_start + kh

            w_start = j * sw
            w_end = w_start + kw

            slice_prev = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(slice_prev, axis=(1, 2))

            elif mode == 'avg':
                A[:, i, j, :] = np.mean(slice_prev, axis=(1, 2))

    return A
