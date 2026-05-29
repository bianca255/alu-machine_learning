#!/usr/bin/env python3
"""Batch normalization module (numpy)."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize an unactivated layer output using batch normalization.

    Args:
        Z: numpy.ndarray of shape (m, n) to normalize.
        gamma: numpy.ndarray of shape (1, n), scale parameters.
        beta: numpy.ndarray of shape (1, n), offset parameters.
        epsilon: small number to avoid division by zero.

    Returns:
        The normalized Z matrix.
    """
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    return gamma * Z_norm + beta
