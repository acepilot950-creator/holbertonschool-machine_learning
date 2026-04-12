#!/usr/bin/env python3
"""Module for performing random crop on images using TensorFlow"""

import tensorflow as tf


def crop_image(image, size):
    """Performs a random crop of an image

    Args:
        image: 3D tf.Tensor containing the image
        size: tuple containing the size of the crop (height, width, channels)

    Returns:
        Cropped image as a tf.Tensor
    """
    return tf.image.random_crop(image, size)
