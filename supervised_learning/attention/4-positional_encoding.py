#!/usr/bin/env python3
"""Positional encoding module for transformers."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate positional encoding vectors for a transformer.

    Args:
        max_seq_len: Maximum sequence length.
        dm: Model depth.

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm)
        containing positional encoding vectors.
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dimensions // 2)) / np.float64(dm)
    )

    angles = positions * angle_rates

    encoding = np.zeros((max_seq_len, dm))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return encoding
