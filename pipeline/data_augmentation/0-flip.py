#!/usr/bin/env python3
"""Module for flipping images horizontally using TensorFlow"""

import tensorflow as tf


def flip_image(image):
    """Flips an image horizontally

    Args:
        image: 3D tf.Tensor containing the image

    Returns:
        Flipped image as a tf.Tensor
    """
    return tf.image.flip_left_right(image)
