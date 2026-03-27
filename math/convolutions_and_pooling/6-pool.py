#!/usr/bin/env python3
"""Performs pooling on images."""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = ((h - kh) // sh) + 1
    output_w = ((w - kw) // sw) + 1
    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            row = i * sh
            col = j * sw
            window = images[:, row:row + kh, col:col + kw, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output
