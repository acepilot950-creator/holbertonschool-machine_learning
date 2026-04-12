#!/usr/bin/env python3
"""Module for rotating images using TensorFlow"""

import tensorflow as tf


def rotate_image(image):
    """Rotates an image by 90 degrees counter-clockwise

    Args:
        image: 3D tf.Tensor containing the image

    Returns:
        Rotated image as a tf.Tensor
    """
    return tf.image.rot90(image, k=1)
