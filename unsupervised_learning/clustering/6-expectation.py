#!/usr/bin/env python3
"""Expectation step for a Gaussian mixture model."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate posterior probabilities and log likelihood."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or not isinstance(pi, np.ndarray) or pi.ndim != 1
            or not isinstance(m, np.ndarray) or m.ndim != 2
            or not isinstance(S, np.ndarray) or S.ndim != 3
            or m.shape[0] != pi.shape[0] or S.shape[0] != pi.shape[0]
            or m.shape[1] != X.shape[1]
            or S.shape[1:] != (X.shape[1], X.shape[1])
            or np.any(pi <= 0)):
        return None, None
    weighted = np.array([pi[i] * pdf(X, m[i], S[i])
                         for i in range(pi.shape[0])])
    total = np.sum(weighted, axis=0)
    if np.any(total <= 0):
        return None, None
    return weighted / total, np.sum(np.log(total))
