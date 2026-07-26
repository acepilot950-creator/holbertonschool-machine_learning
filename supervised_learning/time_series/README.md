# Time Series Forecasting

## Description

This project focuses on preprocessing time series data and using a recurrent neural network to forecast the future price of Bitcoin.

The model uses the previous 24 hours of Bitcoin market data to predict the closing price of the following hour.

The original datasets contain Bitcoin trading information from the Coinbase and Bitstamp exchanges. Each row represents a 60-second time interval.

## Learning Objectives

At the end of this project, I should be able to explain:

* What time series forecasting is
* What a stationary process is
* What a sliding window is
* How to preprocess time series data
* How to create a TensorFlow data pipeline for time series data
* How to perform time series forecasting with RNNs in TensorFlow

## Environment

The project was developed and tested with:

* Ubuntu 20.04 LTS
* Python 3.9
* NumPy 1.25.2
* Pandas 2.2.2
* TensorFlow 2.15
* pycodestyle 2.11.1

## Dataset

The project uses the following Bitcoin datasets:

* Coinbase BTC/USD dataset
* Bitstamp BTC/USD dataset

Each row represents a 60-second trading window and contains:

| Column              | Description                             |
| ------------------- | --------------------------------------- |
| `Timestamp`         | Start time of the interval in Unix time |
| `Open`              | Opening price in USD                    |
| `High`              | Highest price during the interval       |
| `Low`               | Lowest price during the interval        |
| `Close`             | Closing price in USD                    |
| `Volume_(BTC)`      | Amount of BTC traded                    |
| `Volume_(Currency)` | Amount of USD traded                    |
| `Weighted_Price`    | Volume-weighted average price           |

## Project Files

### `preprocess_data.py`

Preprocesses the raw Coinbase and Bitstamp datasets.

The preprocessing includes:

* Loading both CSV datasets
* Removing invalid timestamps and duplicate rows
* Sorting observations chronologically
* Converting Unix timestamps to datetime values
* Aggregating minute-level observations into hourly observations
* Combining data from the two exchanges
* Filling missing hourly observations
* Adding cyclic time features
* Normalizing the features
* Dividing the data chronologically into training and validation sections
* Saving the processed arrays in a compressed NumPy file

The generated file is:

```text
btc_hourly_data.npz
```

### `forecast_btc.py`

Creates, trains, and validates a recurrent neural network for Bitcoin price forecasting.

The script:

* Loads the preprocessed data
* Creates sliding windows containing 24 hours of observations
* Uses each 24-hour window to predict the next hourly closing price
* Creates training and validation pipelines with `tf.data.Dataset`
* Builds a GRU-based neural network
* Uses mean squared error as the loss function
* Uses early stopping and learning-rate reduction
* Saves the best trained model
* Reports validation errors in normalized values and US dollars

## Data Preprocessing

### Hourly aggregation

The original datasets contain one observation per minute. Since the objective is to predict the closing price of the following hour, the data is aggregated into hourly intervals.

The aggregation rules are:

* `Open`: first valid price of the hour
* `High`: highest price of the hour
* `Low`: lowest price of the hour
* `Close`: last valid price of the hour
* `Volume_(BTC)`: total BTC volume during the hour
* `Volume_(Currency)`: total currency volume during the hour
* `Weighted_Price`: BTC-volume-weighted price

### Time features

The raw Unix timestamp is not used directly as a model feature.

Instead, cyclic representations of the hour of the day and day of the week are added:

* `hour_sin`
* `hour_cos`
* `day_sin`
* `day_cos`

These features preserve the cyclic nature of time. For example, hour 23 is close to hour 0.

### Missing values

Missing price values are interpolated and filled using nearby valid observations.

Missing volume values are replaced with zero because no available trade volume is interpreted as no recorded trading activity for that period.

### Normalization

The input features have different numerical scales. For example, Bitcoin prices may be measured in thousands of dollars, while trading volumes may have another range.

Each feature is standardized using:

```text
normalized value = (value - training mean) / training standard deviation
```

The mean and standard deviation are calculated only from the training data to prevent information from the validation period from leaking into the training process.

### Chronological split

Time series observations must not be randomly divided before creating the training and validation sections.

The older observations are used for training, while the newer observations are used for validation.

This simulates the real forecasting problem in which future values are unknown during model training.

## Sliding Window

The model uses a sliding window containing 24 hourly observations.

For example:

```text
Hours 1-24  -> closing price at hour 25
Hours 2-25  -> closing price at hour 26
Hours 3-26  -> closing price at hour 27
```

The input shape used by the recurrent neural network is:

```text
batch size, 24 time steps, number of features
```

## Model Architecture

The model uses a GRU-based recurrent neural network.

Its architecture includes:

* An input layer for 24 hourly time steps
* A GRU layer with 64 units
* A dropout layer
* A GRU layer with 32 units
* A second dropout layer
* A dense hidden layer
* A single output neuron for the predicted closing price

The final output layer has no activation function because Bitcoin price forecasting is a regression problem.

## Loss Function

The model uses mean squared error:

```text
MSE = average of the squared differences between predictions and targets
```

MSE gives larger penalties to predictions that are far from the actual closing price.

Mean absolute error is also tracked as an additional evaluation metric.

## Training

The model uses the Adam optimizer.

The following callbacks are used:

* `ModelCheckpoint` to save the best model
* `EarlyStopping` to stop training when validation loss stops improving
* `ReduceLROnPlateau` to reduce the learning rate when improvement slows down

## Usage

Make the Python scripts executable:

```bash
chmod +x preprocess_data.py
chmod +x forecast_btc.py
```

Run the preprocessing script:

```bash
./preprocess_data.py
```

Custom dataset paths can be provided:

```bash
./preprocess_data.py \
    --coinbase path/to/coinbase.csv \
    --bitstamp path/to/bitstamp.csv \
    --output btc_hourly_data.npz
```

Train and validate the model:

```bash
./forecast_btc.py
```

Custom training parameters can be provided:

```bash
./forecast_btc.py \
    --data btc_hourly_data.npz \
    --model btc_forecast.keras \
    --epochs 30 \
    --batch-size 64
```

## Output Files

The scripts may generate:

```text
btc_hourly_data.npz
btc_forecast.keras
```

`btc_hourly_data.npz` contains the normalized features, target values, normalization statistics, timestamps, and training split index.

`btc_forecast.keras` contains the trained Keras forecasting model.

## Style and Documentation

All Python files:

* Begin with `#!/usr/bin/env python3`
* End with a new line
* Contain module documentation
* Contain function documentation
* Follow `pycodestyle` version 2.11.1

Check code style with:

```bash
pycodestyle preprocess_data.py forecast_btc.py
```

Check module documentation with:

```bash
python3 -c 'print(__import__("preprocess_data").__doc__)'
python3 -c 'print(__import__("forecast_btc").__doc__)'
```

## Repository Structure

```text
.
├── README.md
├── preprocess_data.py
├── forecast_btc.py
├── coinbaseUSD_1-min_data.csv
├── bitstampUSD_1-min_data.csv
├── btc_hourly_data.npz
└── btc_forecast.keras
```

The raw datasets and generated model files may be excluded from version control because of their size.

## Author

This project was completed as part of the Holberton School Machine Learning curriculum.
