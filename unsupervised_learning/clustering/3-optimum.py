#!/usr/bin/env python3
"""Choose K-means cluster counts by variance."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Evaluate K-means for a range of cluster counts."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or not isinstance(kmin, int) or kmin <= 0
            or not isinstance(iterations, int) or iterations <= 0):
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if (not isinstance(kmax, int) or kmax <= kmin
            or kmax > X.shape[0]):
        return None, None
    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        result = kmeans(X, k, iterations)
        if result[0] is None:
            return None, None
        results.append(result)
        variances.append(variance(X, result[0]))
    return results, [variances[0] - value for value in variances]
