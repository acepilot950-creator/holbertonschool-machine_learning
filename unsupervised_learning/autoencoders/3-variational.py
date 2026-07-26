#!/usr/bin/env python3
"""Creates a variational autoencoder."""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder.

    Args:
        input_dims: Dimension of the model input.
        hidden_layers: Number of nodes in each encoder hidden layer.
        latent_dims: Dimension of the latent representation.

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        auto: Complete variational autoencoder model.
    """

    def sampling(args):
        """
        Sample a latent vector using the reparameterization trick.

        Args:
            args: Mean and log variance tensors.

        Returns:
            Sampled latent representation.
        """
        mean, log_variance = args

        epsilon = K.random_normal(
            shape=(K.shape(mean)[0], latent_dims)
        )

        return mean + K.exp(log_variance / 2) * epsilon

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(
            nodes,
            activation="relu"
        )(x)

    mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    log_variance = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    latent = keras.layers.Lambda(sampling)(
        [mean, log_variance]
    )

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[latent, mean, log_variance]
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

    # Complete variational autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded, encoded_mean, encoded_log_variance = encoder(auto_input)
    reconstructed = decoder(encoded)

    auto = keras.Model(
        inputs=auto_input,
        outputs=reconstructed
    )

    # Reconstruction loss
    reconstruction_loss = keras.losses.binary_crossentropy(
        auto_input,
        reconstructed
    )

    reconstruction_loss *= input_dims

    # KL-divergence loss
    kl_loss = 1 + encoded_log_variance
    kl_loss -= K.square(encoded_mean)
    kl_loss -= K.exp(encoded_log_variance)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    # Complete VAE loss
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer="adam")

    return encoder, decoder, auto
