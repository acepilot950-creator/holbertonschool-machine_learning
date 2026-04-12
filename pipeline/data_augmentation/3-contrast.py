#!/usr/bin/env python3
"""Module for randomly adjusting image contrast using TensorFlow"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjusts the contrast of an image

    Args:
        image: 3D tf.Tensor containing the image
        lower: lower bound for contrast factor
        upper: upper bound for contrast factor

    Returns:
        Contrast-adjusted image as a tf.Tensor
    """
    return tf.image.random_contrast(image, lower, upper)
