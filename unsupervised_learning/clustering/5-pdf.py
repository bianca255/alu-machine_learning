#!/usr/bin/env python3
"""Gaussian probability density function."""

import numpy as np


def pdf(X, m, S):
    """Calculate the multivariate Gaussian PDF for each row in X."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or not isinstance(m, np.ndarray) or m.ndim != 1
            or not isinstance(S, np.ndarray) or S.ndim != 2
            or X.shape[1] != m.shape[0]
            or S.shape != (m.shape[0], m.shape[0])):
        return None
    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None
    if det <= 0:
        return None
    delta = X - m
    exponent = -0.5 * np.sum((delta @ inv) * delta, axis=1)
    P = np.exp(exponent) / np.sqrt((2 * np.pi) ** X.shape[1] * det)
    return np.maximum(P, 1e-300)
