#!/usr/bin/env python3
"""Train an RNN model to forecast the next hourly BTC close price."""

import argparse
import os

import numpy as np
import tensorflow as tf


WINDOW_SIZE = 24


def load_preprocessed_data(file_path):
    """Load preprocessed BTC data from a compressed NumPy file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            "Could not find preprocessed data: {}".format(
                file_path
            )
        )

    saved_data = np.load(
        file_path,
        allow_pickle=False,
    )

    required_keys = [
        "features",
        "close",
        "split_index",
        "target_mean",
        "target_standard_deviation",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in saved_data
    ]

    if missing_keys:
        raise ValueError(
            "Missing values in preprocessed file: {}".format(
                ", ".join(missing_keys)
            )
        )

    features = saved_data["features"].astype(
        np.float32
    )
    close_values = saved_data["close"].astype(
        np.float32
    )
    split_index = int(
        saved_data["split_index"]
    )
    target_mean = float(
        saved_data["target_mean"]
    )
    target_standard_deviation = float(
        saved_data["target_standard_deviation"]
    )

    if features.ndim != 2:
        raise ValueError(
            "features must be a two-dimensional array"
        )

    if close_values.ndim != 1:
        raise ValueError(
            "close must be a one-dimensional array"
        )

    if len(features) != len(close_values):
        raise ValueError(
            "features and close must have equal lengths"
        )

    return (
        features,
        close_values,
        split_index,
        target_mean,
        target_standard_deviation,
    )


def create_datasets(
    features,
    close_values,
    split_index,
    batch_size,
):
    """Create training and validation sliding-window datasets."""
    if split_index <= WINDOW_SIZE:
        raise ValueError(
            "Training data must contain more than 24 rows"
        )

    if split_index >= len(features):
        raise ValueError(
            "Validation data must not be empty"
        )

    train_features = features[
        :split_index - 1
    ]
    train_targets = close_values[
        WINDOW_SIZE:split_index
    ]

    validation_start = (
        split_index - WINDOW_SIZE
    )

    validation_features = features[
        validation_start:-1
    ]
    validation_targets = close_values[
        split_index:
    ]

    train_dataset = (
        tf.keras.utils.timeseries_dataset_from_array(
            data=train_features,
            targets=train_targets,
            sequence_length=WINDOW_SIZE,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=True,
            batch_size=batch_size,
        )
    )

    validation_dataset = (
        tf.keras.utils.timeseries_dataset_from_array(
            data=validation_features,
            targets=validation_targets,
            sequence_length=WINDOW_SIZE,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=False,
            batch_size=batch_size,
        )
    )

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )
    validation_dataset = validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return train_dataset, validation_dataset


def build_model(number_of_features):
    """Build and compile a GRU model for BTC price forecasting."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    WINDOW_SIZE,
                    number_of_features,
                )
            ),
            tf.keras.layers.GRU(
                64,
                return_sequences=True,
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),
            tf.keras.layers.Dense(1),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    return model


def calculate_dollar_metrics(
    model,
    validation_dataset,
    target_mean,
    target_standard_deviation,
):
    """Calculate validation MAE and RMSE in original USD units."""
    predictions = model.predict(
        validation_dataset,
        verbose=0,
    ).reshape(-1)

    normalized_targets = []

    for _, targets in validation_dataset:
        normalized_targets.append(
            targets.numpy().reshape(-1)
        )

    normalized_targets = np.concatenate(
        normalized_targets
    )

    predictions_usd = (
        predictions * target_standard_deviation
        + target_mean
    )

    targets_usd = (
        normalized_targets
        * target_standard_deviation
        + target_mean
    )

    errors = predictions_usd - targets_usd

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    return mae, rmse


def train_model(
    data_path,
    model_path,
    epochs,
    batch_size,
):
    """Train and validate the BTC forecasting model."""
    tf.keras.utils.set_random_seed(0)

    (
        features,
        close_values,
        split_index,
        target_mean,
        target_standard_deviation,
    ) = load_preprocessed_data(data_path)

    train_dataset, validation_dataset = (
        create_datasets(
            features=features,
            close_values=close_values,
            split_index=split_index,
            batch_size=batch_size,
        )
    )

    model = build_model(
        number_of_features=features.shape[1]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    validation_results = model.evaluate(
        validation_dataset,
        verbose=0,
    )

    print(
        "Normalized validation MSE: {:.6f}".format(
            validation_results[0]
        )
    )
    print(
        "Normalized validation MAE: {:.6f}".format(
            validation_results[1]
        )
    )

    dollar_mae, dollar_rmse = (
        calculate_dollar_metrics(
            model=model,
            validation_dataset=validation_dataset,
            target_mean=target_mean,
            target_standard_deviation=(
                target_standard_deviation
            ),
        )
    )

    print(
        "Validation MAE in USD: ${:.2f}".format(
            dollar_mae
        )
    )
    print(
        "Validation RMSE in USD: ${:.2f}".format(
            dollar_rmse
        )
    )

    model.save(model_path)

    print("Saved model to:", model_path)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a GRU model for hourly BTC forecasting"
        )
    )

    parser.add_argument(
        "--data",
        default="btc_hourly_data.npz",
        help="Path to the preprocessed NPZ file",
    )

    parser.add_argument(
        "--model",
        default="btc_forecast.keras",
        help="Path used to save the trained model",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )

    return parser.parse_args()


def main():
    """Run BTC forecasting model training."""
    arguments = parse_arguments()

    train_model(
        data_path=arguments.data,
        model_path=arguments.model,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
    )


if __name__ == "__main__":
    main()
