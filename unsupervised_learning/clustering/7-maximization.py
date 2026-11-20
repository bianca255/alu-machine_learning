#!/usr/bin/env python3
"""Maximization step for a Gaussian mixture model."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def maximization(X, g):
    """Calculate updated priors, means, and covariance matrices."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or not isinstance(g, np.ndarray) or g.ndim != 2
            or g.shape[1] != X.shape[0] or g.shape[0] == 0
            or np.any(g < 0)):
        return None, None, None
    weights = np.sum(g, axis=1)
    if np.any(weights <= 0):
        return None, None, None
    pi = weights / X.shape[0]
    m = (g @ X) / weights[:, None]
    S = np.array([
        ((X - m[i]).T * g[i]) @ (X - m[i]) / weights[i]
        for i in range(g.shape[0])
    ])
    return pi, m, S
