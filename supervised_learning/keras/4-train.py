#!/usr/bin/env python3
"""Module that trains a Keras model."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent

    network: model to train
    data: input data
    labels: one-hot labels
    batch_size: size of the batch
    epochs: number of passes through the data
    verbose: determines if output should be printed during training
    shuffle: determines whether to shuffle the data every epoch

    returns: the History object generated after training the model
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
