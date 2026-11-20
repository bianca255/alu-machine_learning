#!/usr/bin/env python3
"""Viterbi algorithm for a hidden Markov model."""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Return the most likely hidden-state sequence."""
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
    probabilities = np.zeros((n, Observation.size))
    parents = np.zeros((n, Observation.size), dtype=int)
    probabilities[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, Observation.size):
        candidates = probabilities[:, t - 1, None] * Transition
        parents[:, t] = np.argmax(candidates, axis=0)
        probabilities[:, t] = (
            candidates[parents[:, t], np.arange(n)]
            * Emission[:, Observation[t]])
    end = np.argmax(probabilities[:, -1])
    probability = probabilities[end, -1]
    path = [end]
    for t in range(Observation.size - 1, 0, -1):
        end = parents[end, t]
        path.append(end)
    path.reverse()
    if probability <= 0:
        return None, None
    return path, probability
