#!/usr/bin/env python3
"""Performs same convolution on grayscale images."""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w) containing
            multiple grayscale images.
        kernel (numpy.ndarray): Array of shape (kh, kw) containing
            the kernel for the convolution.

    Returns:
        numpy.ndarray: The convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            current = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(current * kernel, axis=(1, 2))

    return output
