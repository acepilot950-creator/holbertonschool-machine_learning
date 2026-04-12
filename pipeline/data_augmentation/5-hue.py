#!/usr/bin/env python3
"""Module for adjusting image hue using TensorFlow"""

import tensorflow as tf


def change_hue(image, delta):
    """Changes the hue of an image

    Args:
        image: 3D tf.Tensor containing the image
        delta: amount to change the hue

    Returns:
        Hue-adjusted image as a tf.Tensor
    """
    return tf.image.adjust_hue(image, delta)
