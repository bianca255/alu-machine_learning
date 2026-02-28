#!/usr/bin/env python3
"""Module for calculating the intersection of obtaining data
with various hypothetical probabilities."""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data with the
    various hypothetical probabilities.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array of hypothetical probabilities
        Pr (numpy.ndarray): 1D array of prior beliefs of P

    Returns:
        numpy.ndarray: 1D array of intersection values for each probability
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    from math import factorial
    binom_coeff = factorial(n) / (factorial(x) * factorial(n - x))
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods * Pr
