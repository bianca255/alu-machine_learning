#!/usr/bin/env python3
"""Initialize K-means centroids."""

import numpy as np


def initialize(X, k):
    """Return k centroids sampled uniformly within X's bounds."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                             (k, X.shape[1]))
