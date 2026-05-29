#!/usr/bin/env python3
"""Normalize module."""
import numpy as np


def normalize(X, m, s):
    """Normalize (standardize) a matrix.

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize.
        m: numpy.ndarray of shape (nx,) containing the mean of each feature.
        s: numpy.ndarray of shape (nx,) containing the std of each feature.

    Returns:
        The normalized X matrix.
    """
    return (X - m) / s
