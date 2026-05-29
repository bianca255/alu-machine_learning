#!/usr/bin/env python3
"""Normalization constants module."""
import numpy as np


def normalization_constants(X):
    """Calculate the mean and standard deviation of each feature.

    Args:
        X: numpy.ndarray of shape (m, nx) to normalize.

    Returns:
        Tuple of (mean, std) each of shape (nx,).
    """
    return X.mean(axis=0), X.std(axis=0)
