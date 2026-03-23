#!/usr/bin/env python3
"""Module that configures Adam optimization for a Keras model."""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization for a keras model

    network: the model to optimize
    alpha: learning rate
    beta1: first Adam optimization parameter
    beta2: second Adam optimization parameter
    """

    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
