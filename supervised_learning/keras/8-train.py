#!/usr/bin/env python3
"""Module that trains a Keras model."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent

    network: model to train
    data: input data
    labels: one-hot labels
    batch_size: size of the batch
    epochs: number of passes through the data
    validation_data: data to validate the model with, if not None
    early_stopping: indicates whether early stopping should be used
    patience: patience used for early stopping
    learning_rate_decay: indicates whether learning rate decay should be used
    alpha: initial learning rate
    decay_rate: decay rate
    save_best: indicates whether to save the model after each epoch if it
               is the best
    filepath: file path where the model should be saved
    verbose: determines if output should be printed during training
    shuffle: determines whether to shuffle the data every epoch

    returns: the History object generated after training the model
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(
            K.callbacks.LearningRateScheduler(
                scheduler,
                verbose=1
            )
        )

    if save_best and validation_data is not None:
        callbacks.append(
            K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True
            )
        )

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
