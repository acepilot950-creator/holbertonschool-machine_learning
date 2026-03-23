#!/usr/bin/env python3
"""Module for saving and loading Keras models."""

import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model

    network: the model to save
    filename: the file path where the model should be saved
    """
    network.save(filename)


def load_model(filename):
    """
    loads an entire model

    filename: the file path of the model to load

    Returns: the loaded model
    """
    return K.models.load_model(filename)
