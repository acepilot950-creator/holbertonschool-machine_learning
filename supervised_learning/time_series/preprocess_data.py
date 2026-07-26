#!/usr/bin/env python3
"""Preprocess raw Bitcoin data for time series forecasting."""

import argparse
import os

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
]

PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Weighted_Price",
]

VOLUME_COLUMNS = [
    "Volume_(BTC)",
    "Volume_(Currency)",
]

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


def load_exchange_data(file_path):
    """Load and clean a raw Bitcoin exchange CSV file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            "Could not find dataset: {}".format(file_path)
        )

    data = pd.read_csv(file_path)

    missing_columns = [
        column for column in RAW_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in {}: {}".format(
                file_path,
                ", ".join(missing_columns),
            )
        )

    data = data[RAW_COLUMNS].copy()

    for column in RAW_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(subset=["Timestamp"])
    data["Timestamp"] = data["Timestamp"].astype(np.int64)

    data = data.sort_values("Timestamp")
    data = data.drop_duplicates(
        subset=["Timestamp"],
        keep="last",
    )

    data["Datetime"] = pd.to_datetime(
        data["Timestamp"],
        unit="s",
        utc=True,
    )

    data = data.set_index("Datetime")

    return data


def weighted_price(group):
    """Calculate a BTC-volume-weighted price for one time group."""
    prices = group["Weighted_Price"]
    volumes = group["Volume_(BTC)"]

    valid = prices.notna() & volumes.notna() & (volumes > 0)

    if valid.any():
        numerator = (
            prices.loc[valid] * volumes.loc[valid]
        ).sum()
        denominator = volumes.loc[valid].sum()

        if denominator > 0:
            return numerator / denominator

    return prices.mean()


def resample_exchange(data):
    """Convert one exchange's minute observations into hourly data."""
    price_data = data.resample("1h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume_(BTC)": "sum",
            "Volume_(Currency)": "sum",
        }
    )

    weighted = data.resample("1h").apply(weighted_price)
    weighted.name = "Weighted_Price"

    hourly = price_data.join(weighted)

    return hourly


def combine_exchanges(first_data, second_data):
    """Combine hourly information from two Bitcoin exchanges."""
    first_data = first_data.copy()
    second_data = second_data.copy()

    first_data["Exchange"] = "first"
    second_data["Exchange"] = "second"

    combined = pd.concat(
        [first_data, second_data],
        axis=0,
    )

    combined = combined.sort_index()

    volume_btc = combined["Volume_(BTC)"].fillna(0)
    weighted_value = (
        combined["Weighted_Price"].fillna(0) * volume_btc
    )

    grouped = combined.groupby(combined.index)

    hourly = grouped.agg(
        {
            "Open": "mean",
            "High": "max",
            "Low": "min",
            "Close": "mean",
            "Volume_(BTC)": "sum",
            "Volume_(Currency)": "sum",
        }
    )

    weighted_sum = weighted_value.groupby(
        weighted_value.index
    ).sum()

    volume_sum = volume_btc.groupby(
        volume_btc.index
    ).sum()

    hourly["Weighted_Price"] = np.divide(
        weighted_sum,
        volume_sum,
        out=np.full(
            weighted_sum.shape,
            np.nan,
            dtype=np.float64,
        ),
        where=volume_sum.to_numpy() > 0,
    )

    return hourly.sort_index()


def fill_missing_hours(data):
    """Insert missing hours and fill missing price and volume values."""
    if data.empty:
        raise ValueError("No usable observations were found")

    complete_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="1h",
        tz="UTC",
    )

    data = data.reindex(complete_index)

    data[PRICE_COLUMNS] = data[PRICE_COLUMNS].interpolate(
        method="time",
        limit_direction="both",
    )

    data[PRICE_COLUMNS] = data[PRICE_COLUMNS].ffill()
    data[PRICE_COLUMNS] = data[PRICE_COLUMNS].bfill()

    data[VOLUME_COLUMNS] = data[VOLUME_COLUMNS].fillna(0)

    return data


def add_time_features(data):
    """Add cyclic hour-of-day and day-of-week features."""
    data = data.copy()

    hours = data.index.hour.to_numpy()
    days = data.index.dayofweek.to_numpy()

    data["hour_sin"] = np.sin(
        2 * np.pi * hours / 24
    )
    data["hour_cos"] = np.cos(
        2 * np.pi * hours / 24
    )

    data["day_sin"] = np.sin(
        2 * np.pi * days / 7
    )
    data["day_cos"] = np.cos(
        2 * np.pi * days / 7
    )

    return data


def normalize_data(data, split_index):
    """Normalize features using statistics from training rows only."""
    features = data[FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )

    train_features = features[:split_index]

    means = np.mean(train_features, axis=0)
    standard_deviations = np.std(
        train_features,
        axis=0,
    )

    standard_deviations[
        standard_deviations == 0
    ] = 1.0

    normalized = (
        features - means
    ) / standard_deviations

    close_index = FEATURE_COLUMNS.index("Close")

    close_values = data["Close"].to_numpy(
        dtype=np.float64
    )

    close_mean = means[close_index]
    close_standard_deviation = (
        standard_deviations[close_index]
    )

    normalized_close = (
        close_values - close_mean
    ) / close_standard_deviation

    return (
        normalized,
        normalized_close,
        means,
        standard_deviations,
        close_mean,
        close_standard_deviation,
    )


def preprocess_data(
    coinbase_path,
    bitstamp_path,
    output_path,
    train_ratio,
):
    """Preprocess Coinbase and Bitstamp datasets and save the result."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    coinbase_data = load_exchange_data(
        coinbase_path
    )
    bitstamp_data = load_exchange_data(
        bitstamp_path
    )

    coinbase_hourly = resample_exchange(
        coinbase_data
    )
    bitstamp_hourly = resample_exchange(
        bitstamp_data
    )

    hourly_data = combine_exchanges(
        coinbase_hourly,
        bitstamp_hourly,
    )

    hourly_data = fill_missing_hours(
        hourly_data
    )
    hourly_data = add_time_features(
        hourly_data
    )

    hourly_data = hourly_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    hourly_data = hourly_data.dropna(
        subset=FEATURE_COLUMNS
    )

    split_index = int(
        len(hourly_data) * train_ratio
    )

    if split_index <= 24:
        raise ValueError(
            "The training section must contain more than 24 hours"
        )

    if len(hourly_data) - split_index < 1:
        raise ValueError(
            "The validation section must not be empty"
        )

    (
        normalized_features,
        normalized_close,
        feature_means,
        feature_standard_deviations,
        target_mean,
        target_standard_deviation,
    ) = normalize_data(
        hourly_data,
        split_index,
    )

    timestamps = (
        hourly_data.index.astype("int64")
        // 1_000_000_000
    ).to_numpy()

    np.savez_compressed(
        output_path,
        features=normalized_features.astype(
            np.float32
        ),
        close=normalized_close.astype(
            np.float32
        ),
        timestamps=timestamps.astype(
            np.int64
        ),
        feature_names=np.array(
            FEATURE_COLUMNS
        ),
        feature_means=feature_means.astype(
            np.float32
        ),
        feature_standard_deviations=(
            feature_standard_deviations.astype(
                np.float32
            )
        ),
        target_mean=np.float32(target_mean),
        target_standard_deviation=np.float32(
            target_standard_deviation
        ),
        split_index=np.int64(split_index),
    )

    print("Saved preprocessed data to:", output_path)
    print("Total hourly rows:", len(hourly_data))
    print("Training rows:", split_index)
    print(
        "Validation rows:",
        len(hourly_data) - split_index,
    )
    print("Number of features:", len(FEATURE_COLUMNS))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Coinbase and Bitstamp BTC data"
        )
    )

    parser.add_argument(
        "--coinbase",
        default=(
            "coinbaseUSD_1-min_data_"
            "2014-12-01_to_2019-01-09.csv"
        ),
        help="Path to the Coinbase CSV dataset",
    )

    parser.add_argument(
        "--bitstamp",
        default=(
            "bitstampUSD_1-min_data_"
            "2012-01-01_to_2020-04-22.csv"
        ),
        help="Path to the Bitstamp CSV dataset",
    )

    parser.add_argument(
        "--output",
        default="btc_hourly_data.npz",
        help="Path for the preprocessed NPZ file",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of rows used for training",
    )

    return parser.parse_args()


def main():
    """Run Bitcoin data preprocessing."""
    arguments = parse_arguments()

    preprocess_data(
        coinbase_path=arguments.coinbase,
        bitstamp_path=arguments.bitstamp,
        output_path=arguments.output,
        train_ratio=arguments.train_ratio,
    )


if __name__ == "__main__":
    main()
