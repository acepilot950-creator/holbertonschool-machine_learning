#!/usr/bin/env python3
"""Performs convolution on grayscale images with custom padding and stride."""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w) containing
            multiple grayscale images.
        kernel (numpy.ndarray): Array of shape (kh, kw) containing
            the kernel for the convolution.
        padding (tuple or str): Either a tuple of (ph, pw), 'same',
            or 'valid'.
        stride (tuple): Tuple of (sh, sw) where sh is the stride for
            the height and sw is the stride for the width.

    Returns:
        numpy.ndarray: The convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2))
        pw = int((((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output_h = ((h + (2 * ph) - kh) // sh) + 1
    output_w = ((w + (2 * pw) - kw) // sw) + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            row_start = i * sh
            col_start = j * sw
            current = padded[
                :,
                row_start:row_start + kh,
                col_start:col_start + kw
            ]
            output[:, i, j] = np.sum(current * kernel, axis=(1, 2))

    return output
