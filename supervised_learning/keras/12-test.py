#!/usr/bin/env python3
"""Module for testing a Keras model."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    tests a neural network

    network: the model to test
    data: input data to test the model with
    labels: correct one-hot labels of data
    verbose: determines if output should be printed during testing

    Returns: the loss and accuracy of the model
    """
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
