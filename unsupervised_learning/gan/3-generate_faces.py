#!/usr/bin/env python3
"""Builds convolutional generator and discriminator models."""

import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """Build a convolutional generator and discriminator.

    The generator transforms a latent vector of shape (16,) into
    a grayscale image of shape (16, 16, 1).

    The discriminator receives an image of shape (16, 16, 1)
    and returns a single score.

    Returns:
        tuple: The generator and discriminator Keras models.
    """

    def get_generator():
        """Build and return the generator model."""
        inputs = keras.Input(shape=(16,))

        hidden = keras.layers.Dense(
            2 * 2 * 512,
            activation="tanh"
        )(inputs)

        hidden = keras.layers.Reshape(
            (2, 2, 512)
        )(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)

        hidden = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            padding="same"
        )(hidden)

        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = keras.layers.Activation("tanh")(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)

        hidden = keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding="same"
        )(hidden)

        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = keras.layers.Activation("tanh")(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)

        hidden = keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            padding="same"
        )(hidden)

        hidden = keras.layers.BatchNormalization()(hidden)
        outputs = keras.layers.Activation("tanh")(hidden)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="generator"
        )

    def get_discriminator():
        """Build and return the discriminator model."""
        inputs = keras.Input(shape=(16, 16, 1))

        hidden = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same"
        )(inputs)

        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation("tanh")(hidden)

        hidden = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            padding="same"
        )(hidden)

        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation("tanh")(hidden)

        hidden = keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            padding="same"
        )(hidden)

        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation("tanh")(hidden)

        hidden = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same"
        )(hidden)

        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation("tanh")(hidden)

        hidden = keras.layers.Flatten()(hidden)

        outputs = keras.layers.Dense(
            units=1,
            activation="tanh"
        )(hidden)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="discriminator"
        )

    return get_generator(), get_discriminator()
