# Time Series Forecasting

This project prepares exchange data and trains an RNN to forecast the next hourly Bitcoin closing price.

## Pipeline

1. `preprocess_data.py` reads the Coinbase and Bitstamp CSV files with pandas.
2. The 60-second records are sorted by timestamp and resampled into one-hour observations.
3. Close price, BTC volume, USD volume, and weighted price are retained. Missing and invalid hourly rows are removed.
4. The hourly feature matrix is saved as `preprocessed_data.npz`.
5. `forecast_btc.py` learns normalization statistics from the first 80 percent of the data, creates 24-hour sliding windows with `tf.data.Dataset`, and predicts the following hour's close price.
6. An LSTM followed by dense layers is trained with mean squared error. The model and a training-history graph are saved after training.

## Usage

```text
python3 preprocess_data.py coinbase.csv bitstamp.csv -o preprocessed_data.npz
python3 forecast_btc.py preprocessed_data.npz -m btc_forecaster.keras -p training_history.png
```

The split is chronological rather than random so validation represents future observations. The 24-hour context gives the model one complete daily cycle while the hourly aggregation removes much of the noise in the original minute-level data.

## Blog deliverable

The project specification also requires publishing an English blog post with an image, stationarity and seasonality discussion, preprocessing and `tf.data.Dataset` details, model results, graphs, conclusion, and links to this repository. That external publication must be completed manually on Medium or LinkedIn and cannot be generated or published from this repository.
