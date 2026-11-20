#!/usr/bin/env python3
"""Regular Markov chain steady-state probabilities."""

import numpy as np


def regular(P):
    """Return the steady-state vector for a regular transition matrix."""
    if (not isinstance(P, np.ndarray) or P.ndim != 2
            or P.shape[0] != P.shape[1] or P.shape[0] == 0
            or not np.issubdtype(P.dtype, np.number)
            or not np.all(np.isfinite(P)) or np.any(P < 0)
            or not np.allclose(P.sum(axis=1), 1)):
        return None
    n = P.shape[0]
    power = P.copy()
    for _ in range(n):
        if np.all(power > 0):
            break
        power = power.dot(P)
    else:
        if not np.all(power > 0):
            return None
    A = np.vstack((P.T - np.eye(n), np.ones(n)))
    b = np.zeros(n + 1)
    b[-1] = 1
    try:
        result = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if (not np.all(np.isfinite(result)) or np.any(result < -1e-10)
            or not np.isclose(result.sum(), 1)):
        return None
    result = np.maximum(result, 0)
    result = result / result.sum()
    return result.reshape((1, n))
