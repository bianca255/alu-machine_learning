#!/usr/bin/env python3
"""Maximization step of the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step of the EM algorithm for a GMM.

    X: numpy.ndarray of shape (n, d) containing the data set
    g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns: pi, m, S, or None, None, None on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or X.shape[1] == 0 or not np.issubdtype(X.dtype, np.number)
            or not np.all(np.isfinite(X))):
        return None, None, None
    if (not isinstance(g, np.ndarray) or g.ndim != 2
            or g.shape[0] == 0 or g.shape[1] != X.shape[0]
            or not np.issubdtype(g.dtype, np.number)
            or not np.all(np.isfinite(g)) or np.any(g < 0)):
        return None, None, None
    if not np.all(np.isclose(np.sum(g, axis=0), 1)):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    N_k = np.sum(g, axis=1)
    if np.any(N_k <= 0):
        return None, None, None

    pi = N_k / n
    m = np.matmul(g, X) / N_k[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted = g[i][:, np.newaxis] * diff
        S[i] = np.matmul(weighted.T, diff) / N_k[i]
    return pi, m, S
