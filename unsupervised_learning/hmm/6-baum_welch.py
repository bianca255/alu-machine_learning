#!/usr/bin/env python3
"""Baum-Welch training algorithm for a hidden Markov model."""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """Return the converged transition and emission matrices."""
    if (not isinstance(Observations, np.ndarray)
            or Observations.ndim != 1 or Observations.size == 0
            or not isinstance(Transition, np.ndarray)
            or Transition.ndim != 2
            or not isinstance(Emission, np.ndarray) or Emission.ndim != 2
            or not isinstance(Initial, np.ndarray) or Initial.ndim != 2
            or Initial.shape[1] != 1
            or not np.issubdtype(Observations.dtype, np.integer)
            or np.any(Observations < 0)
            or np.max(Observations) >= Emission.shape[1]
            or not isinstance(iterations, int)
            or isinstance(iterations, bool) or iterations <= 0):
        return None, None
    m = Emission.shape[0]
    n = Emission.shape[1]
    if (Transition.shape != (m, m) or Initial.shape[0] != m
            or not np.issubdtype(Transition.dtype, np.number)
            or not np.issubdtype(Emission.dtype, np.number)
            or not np.issubdtype(Initial.dtype, np.number)
            or not np.all(np.isfinite(Transition))
            or not np.all(np.isfinite(Emission))
            or not np.all(np.isfinite(Initial))
            or np.any(Transition < 0) or np.any(Emission < 0)
            or np.any(Initial < 0)):
        return None, None
    Transition = Transition.copy()
    Emission = Emission.copy()
    Initial = Initial.copy()
    t_count = Observations.size
    for _ in range(iterations):
        alpha = np.zeros((m, t_count))
        alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, t_count):
            alpha[:, t] = (
                Transition.T @ alpha[:, t - 1]
            ) * Emission[:, Observations[t]]
        likelihood = np.sum(alpha[:, -1])
        if likelihood <= 0:
            return None, None
        beta = np.zeros((m, t_count))
        beta[:, -1] = 1
        for t in range(t_count - 2, -1, -1):
            beta[:, t] = Transition @ (
                Emission[:, Observations[t + 1]] * beta[:, t + 1])
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=0)
        xi = np.zeros((m, m, t_count - 1))
        for t in range(t_count - 1):
            denominator = (
                alpha[:, t, None]
                * Transition
                * Emission[None, :, Observations[t + 1]]
                * beta[None, :, t + 1]
            ).sum()
            if denominator > 0:
                xi[:, :, t] = (
                    alpha[:, t, None]
                    * Transition
                    * Emission[None, :, Observations[t + 1]]
                    * beta[None, :, t + 1]
                ) / denominator
        transition_denominator = np.sum(gamma[:, :-1], axis=1)
        transition_numerator = np.sum(xi, axis=2)
        new_transition = np.divide(
            transition_numerator,
            transition_denominator[:, None],
            out=np.zeros_like(Transition),
            where=transition_denominator[:, None] > 0)
        new_emission = np.zeros_like(Emission)
        for k in range(n):
            mask = Observations == k
            new_emission[:, k] = np.sum(gamma[:, mask], axis=1)
        emission_denominator = np.sum(gamma, axis=1)
        new_emission = np.divide(
            new_emission,
            emission_denominator[:, None],
            out=np.zeros_like(Emission),
            where=emission_denominator[:, None] > 0)
        if (np.allclose(new_transition, Transition)
                and np.allclose(new_emission, Emission)):
            return new_transition, new_emission
        Transition = new_transition
        Emission = new_emission
    return Transition, Emission
