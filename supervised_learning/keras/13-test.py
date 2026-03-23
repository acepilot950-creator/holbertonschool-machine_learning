
#!/usr/bin/env python3
"""Module for making predictions with a neural network."""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network

    network: the model to use for prediction
    data: the input data to make predictions with
    verbose: determines if output should be printed during prediction

    Returns: the prediction for the data
    """
    return network.predict(data, verbose=verbose)
