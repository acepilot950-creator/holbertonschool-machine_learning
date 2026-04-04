#!/usr/bin/env python3
"""Convolutional forward propagation for a convolutional layer."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same",
                 stride=(1, 1)):
    """Performs forward propagation over a convolutional layer.

    Args:
        A_prev (numpy.ndarray): Output of previous layer of shape
            (m, h_prev, w_prev, c_prev).
        W (numpy.ndarray): Kernels for convolution of shape
            (kh, kw, c_prev, c_new).
        b (numpy.ndarray): Biases of shape (1, 1, 1, c_new).
        activation (function): Activation function applied to
            the convolution output.
        padding (str): Type of padding, either "same" or "valid".
        stride (tuple): Strides for the convolution as (sh, sw).

    Returns:
        numpy.ndarray: Activated output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil((((h_prev - 1) * sh) + kh - h_prev) / 2))
        pw = int(np.ceil((((w_prev - 1) * sw) + kw - w_prev) / 2))
    else:
        ph = 0
        pw = 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    h_new = int(((h_prev + (2 * ph) - kh) / sh) + 1)
    w_new = int(((w_prev + (2 * pw) - kw) / sw) + 1)

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            current = A_prev_pad[:, h_start:h_end, w_start:w_end, :]

            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    current * W[:, :, :, k],
                    axis=(1, 2, 3)
                ) + b[0, 0, 0, k]

    return activation(Z)
