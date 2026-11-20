#!/usr/bin/env python3
"""Initialize a Gaussian mixture model."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Return uniform priors, K-means means, and identity covariances."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or not isinstance(k, int) or k <= 0 or k > X.shape[0]):
        return None, None, None
    C, _ = kmeans(X, k)
    if C is None:
        return None, None, None
    return (np.full(k, 1 / k), C, np.tile(np.eye(X.shape[1]), (k, 1, 1)))
