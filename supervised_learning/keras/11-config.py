#!/usr/bin/env python3
"""Module for saving and loading a model configuration."""

import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model's configuration in JSON format

    network: the model whose configuration should be saved
    filename: the file path where the configuration should be saved
    """
    config = network.to_json()

    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """
    loads a model with a specific configuration

    filename: the file path of the configuration file

    Returns: the loaded model
    """
    with open(filename, 'r') as f:
        config = f.read()

    return K.models.model_from_json(config)
