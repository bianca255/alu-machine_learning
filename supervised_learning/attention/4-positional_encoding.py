#!/usr/bin/env python3
"""Sinusoidal positional encodings for Transformer inputs."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Return positional encodings with shape ``(max_seq_len, dm)``."""
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(dm))
    angles = positions * angle_rates
    encoding = np.zeros((max_seq_len, dm))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding
