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

    def __init__(self, style_image, content_image, alpha=1e4,
                 beta=1, var=10):
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

        if (
            not isinstance(var, (int, float))
            or isinstance(var, bool)
            or var < 0
        ):
            raise TypeError('var must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Resize an image and scale its pixels to the range [0, 1]."""
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
            """Replace max-pooling layers with average-pooling layers."""
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

        gram = gram / tf.cast(height * width, tf.float32)

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

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for one layer."""
        if (
            not isinstance(style_output, (tf.Tensor, tf.Variable))
            or len(style_output.shape) != 4
        ):
            raise TypeError('style_output must be a tensor of rank 4')

        channels = style_output.shape[-1]

        if (
            not isinstance(gram_target, (tf.Tensor, tf.Variable))
            or len(gram_target.shape) != 3
            or gram_target.shape[0] != 1
            or gram_target.shape[1] != channels
            or gram_target.shape[2] != channels
        ):
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}]'.format(
                    channels, channels
                )
            )

        gram_style = self.gram_matrix(style_output)

        layer_cost = tf.reduce_sum(
            tf.square(gram_style - gram_target)
        )

        layer_cost = layer_cost / tf.cast(
            channels ** 2,
            tf.float32
        )

        return layer_cost

    def style_cost(self, style_outputs):
        """Calculate the total style cost for the generated image."""
        length = len(self.style_layers)

        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError(
                'style_outputs must be a list with a length of {}'.format(
                    length
                )
            )

        weight = 1 / length
        cost = 0

        for output, target in zip(
            style_outputs,
            self.gram_style_features
        ):
            cost += weight * self.layer_style_cost(output, target)

        return cost

    def content_cost(self, content_output):
        """Calculate the content cost for the generated image."""
        target_shape = self.content_feature.shape

        if (
            not isinstance(content_output, (tf.Tensor, tf.Variable))
            or content_output.shape != target_shape
        ):
            raise TypeError(
                'content_output must be a tensor of shape {}'.format(
                    target_shape
                )
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

    @staticmethod
    def variational_cost(generated_image):
        """Calculate the variational cost of a generated image."""
        return tf.reduce_sum(
            tf.image.total_variation(generated_image)
        )

    def total_cost(self, generated_image):
        """Calculate total, content, style, and variational costs."""
        target_shape = self.content_image.shape

        if (
            not isinstance(generated_image, (tf.Tensor, tf.Variable))
            or generated_image.shape != target_shape
        ):
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(
                    target_shape
                )
            )

        preprocessed_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )

        outputs = self.model(preprocessed_image)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        j_content = self.content_cost(content_output)
        j_style = self.style_cost(style_outputs)
        j_var = self.variational_cost(generated_image)

        j_total = (
            self.alpha * j_content
            + self.beta * j_style
            + self.var * j_var
        )

        return j_total, j_content, j_style, j_var

    def compute_grads(self, generated_image):
        """Calculate gradients and costs for a generated image."""
        target_shape = self.content_image.shape

        if (
            not isinstance(generated_image, (tf.Tensor, tf.Variable))
            or generated_image.shape != target_shape
        ):
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(
                    target_shape
                )
            )

        with tf.GradientTape() as tape:
            if not isinstance(generated_image, tf.Variable):
                tape.watch(generated_image)

            costs = self.total_cost(generated_image)
            j_total, j_content, j_style, j_var = costs

        gradients = tape.gradient(j_total, generated_image)

        return gradients, j_total, j_content, j_style, j_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generate the neural style transferred image."""
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError('iterations must be an integer')

        if iterations <= 0:
            raise ValueError('iterations must be positive')

        if step is not None:
            if not isinstance(step, int) or isinstance(step, bool):
                raise TypeError('step must be an integer')

            if step <= 0 or step >= iterations:
                raise ValueError(
                    'iterations must be positive and less than iterations'
                )

        if (
            not isinstance(lr, (int, float))
            or isinstance(lr, bool)
        ):
            raise TypeError('lr must be a number')

        if lr <= 0:
            raise ValueError('lr must be positive')

        if not isinstance(beta1, float):
            raise TypeError('beta1 must be a float')

        if beta1 < 0 or beta1 > 1:
            raise ValueError('beta1 must be in the range [0, 1]')

        if not isinstance(beta2, float):
            raise TypeError('beta2 must be a float')

        if beta2 < 0 or beta2 > 1:
            raise ValueError('beta2 must be in the range [0, 1]')

        optimizer = tf.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        generated_image = tf.Variable(self.content_image)
        best_image = generated_image[0].numpy()
        best_cost = np.inf

        for i in range(iterations + 1):
            results = self.compute_grads(generated_image)
            gradients = results[0]
            j_total = results[1]
            j_content = results[2]
            j_style = results[3]
            j_var = results[4]

            current_cost = j_total.numpy()

            if current_cost < best_cost:
                best_cost = current_cost
                best_image = generated_image[0].numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    'Cost at iteration {}: {}, content {}, '
                    'style {}, var {}'.format(
                        i,
                        j_total.numpy(),
                        j_content.numpy(),
                        j_style.numpy(),
                        j_var.numpy()
                    )
                )

            if i != iterations:
                optimizer.apply_gradients(
                    [(gradients, generated_image)]
                )

                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0)
                )

        return best_image, best_cost
