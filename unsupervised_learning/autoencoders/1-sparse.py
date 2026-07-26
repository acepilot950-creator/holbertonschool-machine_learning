#!/usr/bin/env python3
"""Creates a sparse autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder.

    Args:
        input_dims: Dimension of the model input.
        hidden_layers: Number of nodes in each hidden encoder layer.
        latent_dims: Dimension of the latent representation.
        lambtha: L1 activity regularization parameter.

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        auto: Complete sparse autoencoder model.
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(
            nodes,
            activation="relu"
        )(x)

    latent = keras.layers.Dense(
        latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=latent
    )

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(
            nodes,
            activation="relu"
        )(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    # Complete sparse autoencoder
    auto_input = keras.Input(shape=(input_dims,))
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
