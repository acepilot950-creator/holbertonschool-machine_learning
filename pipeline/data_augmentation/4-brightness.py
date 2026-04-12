#!/usr/bin/env python3
"""Module for randomly adjusting image brightness using TensorFlow"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image

    Args:
        image: 3D tf.Tensor containing the image
        max_delta: maximum amount to change brightness

    Returns:
        Brightness-adjusted image as a tf.Tensor
    """
    return tf.image.random_brightness(image, max_delta)
