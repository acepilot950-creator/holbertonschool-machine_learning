#!/usr/bin/env python3
"""Module for saving and loading model weights."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    saves a model's weights

    network: the model whose weights should be saved
    filename: the path of the file that the weights should be saved to
    save_format: the format in which the weights should be saved
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    loads a model's weights

    network: the model to which the weights should be loaded
    filename: the path of the file that the weights should be loaded from
    """
    network.load_weights(filename)
