#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means clustering."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or not isinstance(k, int) or k <= 0 or k > X.shape[0]
            or not isinstance(iterations, int) or iterations <= 0):
        return None, None
    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                          (k, X.shape[1]))
    for _ in range(iterations):
        distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        clss = np.argmin(distances, axis=1)
        new_C = np.array([
            X[clss == i].mean(axis=0) if np.any(clss == i)
            else np.random.uniform(np.min(X, axis=0), np.max(X, axis=0))
            for i in range(k)
        ])
        if np.array_equal(C, new_C):
            return new_C, clss
        C = new_C
    distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    return C, np.argmin(distances, axis=1)
