#!/usr/bin/env python3
"""One-hot encode function."""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix.

    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels
        classes: Maximum number of classes found in Y

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < 1:
        return None
    if np.max(Y) >= classes or np.min(Y) < 0:
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1

    return one_hot
