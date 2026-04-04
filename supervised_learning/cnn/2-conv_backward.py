#!/usr/bin/env python3
"""Back propagation over a convolutional layer."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer.

    Args:
        dZ (numpy.ndarray): Partial derivatives with respect to the
            unactivated output of the convolutional layer, with shape
            (m, h_new, w_new, c_new).
        A_prev (numpy.ndarray): Output of the previous layer, with shape
            (m, h_prev, w_prev, c_prev).
        W (numpy.ndarray): Kernels for the convolution, with shape
            (kh, kw, c_prev, c_new).
        b (numpy.ndarray): Biases applied to the convolution, with shape
            (1, 1, 1, c_new).
        padding (str): Type of padding used, either "same" or "valid".
        stride (tuple): Strides for the convolution as (sh, sw).

    Returns:
        tuple: dA_prev, dW, db
            dA_prev is the partial derivatives with respect to A_prev
            dW is the partial derivatives with respect to W
            db is the partial derivatives with respect to b
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
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
    dA_prev_pad = np.pad(
        np.zeros_like(A_prev),
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        a_prev = A_prev_pad[i]
        da_prev = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for c in range(c_new):
                    a_slice = a_prev[
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ]

                    da_prev[
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ] += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        dA_prev_pad[i] = da_prev

    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
