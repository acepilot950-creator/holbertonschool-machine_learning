#!/usr/bin/env python3
"""Creates a convolutional autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims: Tuple containing the dimensions of the model input.
        filters: List containing the number of filters for each
            convolutional layer in the encoder.
        latent_dims: Tuple containing the dimensions of the latent space.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The complete convolutional autoencoder model.
    """
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for number_filters in filters:
        x = keras.layers.Conv2D(
            filters=number_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(x)

        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="same"
        )(x)

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=x
    )

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    reversed_filters = list(reversed(filters))

    for index, number_filters in enumerate(reversed_filters):
        if index == len(reversed_filters) - 1:
            padding = "valid"
        else:
            padding = "same"

        x = keras.layers.Conv2D(
            filters=number_filters,
            kernel_size=(3, 3),
            padding=padding,
            activation="relu"
        )(x)

        x = keras.layers.UpSampling2D(
            size=(2, 2)
        )(x)

    decoder_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding="same",
        activation="sigmoid"
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    # Complete autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    reconstructed = decoder(encoded)

    auto = keras.Model(
        inputs=auto_input,
        outputs=reconstructed
    )

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
