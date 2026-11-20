#!/usr/bin/env python3
"""Backward algorithm for a hidden Markov model."""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Calculate observation likelihood and backward probabilities."""
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1
            or Observation.size == 0
            or not isinstance(Emission, np.ndarray) or Emission.ndim != 2
            or not isinstance(Transition, np.ndarray)
            or Transition.ndim != 2
            or not isinstance(Initial, np.ndarray) or Initial.ndim != 2
            or Initial.shape[1] != 1
            or not np.issubdtype(Observation.dtype, np.integer)
            or np.any(Observation < 0)
            or np.max(Observation) >= Emission.shape[1]):
        return None, None
    n, m = Emission.shape
    if (Transition.shape != (n, n) or Initial.shape[0] != n
            or not np.issubdtype(Emission.dtype, np.number)
            or not np.issubdtype(Transition.dtype, np.number)
            or not np.issubdtype(Initial.dtype, np.number)
            or not np.all(np.isfinite(Emission))
            or not np.all(np.isfinite(Transition))
            or not np.all(np.isfinite(Initial))
            or np.any(Emission < 0) or np.any(Transition < 0)
            or np.any(Initial < 0)):
        return None, None
    B = np.zeros((n, Observation.size))
    B[:, -1] = 1
    for t in range(Observation.size - 2, -1, -1):
        B[:, t] = Transition @ (
            Emission[:, Observation[t + 1]] * B[:, t + 1])
    probability = np.sum(
        Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    if probability <= 0:
        return None, None
    return probability, B
