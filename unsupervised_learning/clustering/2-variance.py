#!/usr/bin/env python3
"""K-means variance."""

import numpy as np


def variance(X, C):
    """Calculate total intra-cluster variance."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or not isinstance(C, np.ndarray) or C.ndim != 2
            or X.shape[1] != C.shape[1] or C.shape[0] == 0):
        return None
    distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    return np.sum(np.min(distances, axis=1) ** 2)
