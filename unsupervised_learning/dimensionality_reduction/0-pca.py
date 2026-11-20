#!/usr/bin/env python3
"""Principal component analysis with variance selection."""

import numpy as np


def pca(X, var=0.95):
    """Return eigenvector weights retaining at least var variance."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or X.shape[1] == 0 or not np.issubdtype(X.dtype, np.number)
            or not np.all(np.isfinite(X)) or isinstance(var, bool)
            or not isinstance(var, (int, float)) or var <= 0 or var > 1):
        return None
    covariance = np.matmul(X.T, X) / X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    total = np.sum(eigenvalues)
    if total <= 0:
        return None
    cumulative = np.cumsum(eigenvalues) / total
    ndim = np.searchsorted(cumulative, var) + 1
    return eigenvectors[:, :ndim]
