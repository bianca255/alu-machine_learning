#!/usr/bin/env python3
"""Principal component analysis for a fixed output dimensionality."""

import numpy as np


def pca(X, ndim):
    """Return X projected onto its first ndim principal components."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or X.shape[1] == 0 or not np.issubdtype(X.dtype, np.number)
            or not np.all(np.isfinite(X)) or isinstance(ndim, bool)
            or not isinstance(ndim, int) or ndim <= 0
            or ndim > X.shape[1]):
        return None
    X_centered = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return np.matmul(X_centered, Vt[:ndim].T)
