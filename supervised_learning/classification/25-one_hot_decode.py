#!/usr/bin/env python3
"""One-hot decode function."""

import numpy as np


def one_hot_decode(one_hot):
    """
    Convert a one-hot matrix into a vector of labels.

    Args:
        one_hot: One-hot encoded numpy.ndarray with shape (classes, m)

    Returns:
        A numpy.ndarray with shape (m,) containing numeric labels, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    return np.argmax(one_hot, axis=0)
