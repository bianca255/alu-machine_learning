#!/usr/bin/env python3
"""Prepare Coinbase and Bitstamp data for hourly BTC forecasting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["close", "volume_btc", "volume_currency", "weighted_price"]


def _normalise_columns(frame):
    """Rename common exchange column spellings to canonical names."""
    aliases = {
        "timestamp": "timestamp",
        "time": "timestamp",
        "date": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume_btc": "volume_btc",
        "volume btc": "volume_btc",
        "volume_(btc)": "volume_btc",
        "volume_currency": "volume_currency",
        "volume usd": "volume_currency",
        "volume_(currency)": "volume_currency",
        "weighted_price": "weighted_price",
        "weighted price": "weighted_price",
    }
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace("-", "_")
        if key in aliases:
            renamed[column] = aliases[key]
    return frame.rename(columns=renamed)


def load_exchange_data(path):
    """Load one exchange CSV and return its canonical time-indexed data."""
    frame = _normalise_columns(pd.read_csv(path))
    required = {"timestamp", "close", "volume_btc", "volume_currency",
                "weighted_price"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Missing columns in {}: {}".format(
            path, ", ".join(sorted(missing))))
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    if timestamps.notna().all():
        unit = "ms" if timestamps.max() > 10**11 else "s"
        frame["timestamp"] = pd.to_datetime(timestamps, unit=unit, utc=True)
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame[["close", "volume_btc", "volume_currency",
                  "weighted_price"]].apply(pd.to_numeric, errors="coerce")


def preprocess_data(paths, output_path):
    """Aggregate exchange data to hourly rows and save a compressed NumPy file.

    Args:
        paths: Iterable of Coinbase and Bitstamp CSV paths.
        output_path: Destination `.npz` path.

    Returns:
        The hourly feature DataFrame written to ``output_path``.
    """
    frames = [load_exchange_data(path) for path in paths]
    combined = pd.concat(frames).sort_index()
    hourly = combined.resample("1h").agg({
        "close": "last",
        "volume_btc": "sum",
        "volume_currency": "sum",
        "weighted_price": "mean",
    }).dropna()
    hourly = hourly[(hourly[FEATURES] >= 0).all(axis=1)]
    if hourly.empty:
        raise ValueError("No valid hourly observations were produced")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, data=hourly[FEATURES].to_numpy(
        dtype=np.float32), timestamps=hourly.index.astype(str).to_numpy(),
        features=np.array(FEATURES))
    return hourly


def main():
    """Parse command-line arguments and preprocess the exchange files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coinbase", help="Path to the Coinbase CSV")
    parser.add_argument("bitstamp", help="Path to the Bitstamp CSV")
    parser.add_argument("-o", "--output", default="preprocessed_data.npz",
                        help="Output NumPy archive")
    args = parser.parse_args()
    hourly = preprocess_data([args.coinbase, args.bitstamp], args.output)
    message = "Saved {} hourly observations to {}".format(
        len(hourly), args.output)
    print(message)


if __name__ == "__main__":
    main()
