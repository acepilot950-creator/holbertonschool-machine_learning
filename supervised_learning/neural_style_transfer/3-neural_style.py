#!/usr/bin/env python3
"""Module for neural style transfer."""

import numpy as np
import tensorflow as tf


class NST:
    """Perform tasks for neural style transfer."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize a neural style transfer instance."""
        if (
            not isinstance(style_image, np.ndarray)
            or style_image.ndim != 3
            or style_image.shape[2] != 3
        ):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if (
            not isinstance(content_image, np.ndarray)
            or content_image.ndim != 3
            or content_image.shape[2] != 3
        ):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or alpha < 0
        ):
            raise TypeError('alpha must be a non-negative number')

        if (
            not isinstance(beta, (int, float))
            or isinstance(beta, bool)
            or beta < 0
        ):
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Resize an image and scale its pixel values between 0 and 1."""
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )

        height, width, _ = image.shape
        scale = 512 / max(height, width)

        new_height = int(height * scale)
        new_width = int(width * scale)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        image = tf.image.resize(
            image,
            (new_height, new_width),
            method='bicubic'
        )

        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def load_model(self):
        """Load the VGG19 model used to extract image features."""
        base_model = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        def replace_pooling(layer):
            """Replace max pooling layers with average pooling layers."""
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                config = layer.get_config()
                return tf.keras.layers.AveragePooling2D(**config)

            return layer.__class__.from_config(layer.get_config())

        model = tf.keras.models.clone_model(
            base_model,
            clone_function=replace_pooling
        )

        model.set_weights(base_model.get_weights())

        outputs = [
            model.get_layer(layer_name).output
            for layer_name in self.style_layers
        ]

        outputs.append(
            model.get_layer(self.content_layer).output
        )

        self.model = tf.keras.Model(
            inputs=model.input,
            outputs=outputs
        )

        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the Gram matrix of a layer output."""
        if (
            not isinstance(input_layer, (tf.Tensor, tf.Variable))
            or len(input_layer.shape) != 4
        ):
            raise TypeError('input_layer must be a tensor of rank 4')

        shape = tf.shape(input_layer)
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        features = tf.reshape(
            input_layer,
            (shape[0], height * width, channels)
        )

        gram = tf.matmul(
            features,
            features,
            transpose_a=True
        )

        gram /= tf.cast(height * width, tf.float32)

        return gram

    def generate_features(self):
        """Extract the style and content features from the images."""
        style_image = self.style_image * 255.0
        content_image = self.content_image * 255.0

        style_image = tf.keras.applications.vgg19.preprocess_input(
            style_image
        )
        content_image = tf.keras.applications.vgg19.preprocess_input(
            content_image
        )

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        self.gram_style_features = [
            self.gram_matrix(output)
            for output in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]
