#!/usr/bin/env python3
"""Train an LSTM model to forecast the next hourly BTC close price."""

import argparse

import numpy as np
import tensorflow as tf


WINDOW = 24
FEATURE_COUNT = 4


def load_data(path):
    """Load the feature matrix from a preprocessing archive."""
    with np.load(path, allow_pickle=False) as archive:
        data = archive["data"].astype(np.float32)
    if data.ndim != 2 or data.shape[1] != FEATURE_COUNT:
        raise ValueError("Expected four hourly features in the data archive")
    return data


def scale_data(data, split_index):
    """Standardize data using statistics learned from the training period."""
    mean = data[:split_index].mean(axis=0)
    deviation = data[:split_index].std(axis=0)
    deviation[deviation == 0] = 1
    return (data - mean) / deviation, mean, deviation


def make_dataset(data, window=WINDOW, batch_size=32, shuffle=False):
    """Create `(past_window, next_close)` samples with `tf.data`."""
    if len(data) <= window:
        raise ValueError("At least window + 1 observations are required")
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(window + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda item: item.batch(window + 1))
    dataset = dataset.map(lambda item: (item[:-1], item[-1, 0]))
    if shuffle:
        dataset = dataset.shuffle(1024, seed=0)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(window=WINDOW, features=FEATURE_COUNT):
    """Build and compile the LSTM forecasting model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, features)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def plot_history(history, output_path):
    """Save training and validation loss curves as an image."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(history.history["loss"], label="training loss")
    axis.plot(history.history["val_loss"], label="validation loss")
    axis.set(xlabel="Epoch", ylabel="MSE", title="BTC forecast training")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def train_forecaster(data_path, model_path, plot_path, epochs, batch_size):
    """Train, evaluate, save, and return an hourly BTC forecaster."""
    data = load_data(data_path)
    split = max(WINDOW + 1, int(len(data) * 0.8))
    scaled, mean, deviation = scale_data(data, split)
    train = make_dataset(scaled[:split], batch_size=batch_size, shuffle=True)
    validation_start = max(0, split - WINDOW)
    validation = make_dataset(scaled[validation_start:],
                              batch_size=batch_size)
    model = build_model()
    history = model.fit(train, validation_data=validation, epochs=epochs)
    model.save(model_path)
    plot_history(history, plot_path)
    metrics = model.evaluate(validation, verbose=0)
    print("Validation MSE: {:.6f}".format(metrics[0]))
    return model, mean, deviation


def main():
    """Parse arguments and train the BTC forecasting model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Preprocessed `.npz` input")
    parser.add_argument("-m", "--model", default="btc_forecaster.keras")
    parser.add_argument("-p", "--plot", default="training_history.png")
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_forecaster(args.data, args.model, args.plot, args.epochs,
                     args.batch_size)


if __name__ == "__main__":
    main()
