#!/usr/bin/env python3
"""Absorbing Markov chain detection."""

import numpy as np


def absorbing(P):
    """Return True if every state can reach an absorbing state."""
    if (not isinstance(P, np.ndarray) or P.ndim != 2
            or P.shape[0] != P.shape[1] or P.shape[0] == 0
            or not np.issubdtype(P.dtype, np.number)
            or not np.all(np.isfinite(P)) or np.any(P < 0)
            or not np.allclose(P.sum(axis=1), 1)):
        return False
    n = P.shape[0]
    absorbing_states = np.where(np.isclose(P.diagonal(), 1))[0]
    if absorbing_states.size == 0:
        return False
    reachable = P > 0
    for k in range(n):
        reachable |= reachable[:, k:k + 1] & reachable[k:k + 1, :]
    return bool(np.all(reachable[np.arange(n)[:, None],
                                 absorbing_states].any(axis=1)))
