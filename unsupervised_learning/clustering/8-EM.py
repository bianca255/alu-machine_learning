#!/usr/bin/env python3
"""Expectation maximization for a Gaussian mixture model."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Fit a Gaussian mixture model with EM."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or not isinstance(k, int) or k <= 0 or k > X.shape[0]
            or not isinstance(iterations, int) or iterations <= 0
            or not isinstance(tol, (int, float)) or tol < 0
            or not isinstance(verbose, bool)):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    previous = None
    for i in range(iterations):
        g, likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, likelihood))
        if previous is not None and abs(likelihood - previous) <= tol:
            return pi, m, S, g, likelihood
        updated = maximization(X, g)
        if updated[0] is None:
            return None, None, None, None, None
        pi, m, S = updated
        previous = likelihood
    return pi, m, S, g, likelihood
