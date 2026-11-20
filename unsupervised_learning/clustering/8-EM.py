#!/usr/bin/env python3
"""Expectation Maximization for a GMM."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform the expectation maximization algorithm for a GMM.

    X: numpy.ndarray of shape (n, d) containing the data set
    k: positive integer, number of clusters
    iterations: positive integer, maximum number of iterations
    tol: non-negative float, tolerance of the log likelihood
    verbose: boolean, whether to print log likelihood information

    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)
    if g is None or log_likelihood is None:
        return None, None, None, None, None

    l_prev = 0
    i = 0
    for i in range(iterations + 1):
        if i > 0:
            pi, m, S = maximization(X, g)
            if pi is None or m is None or S is None:
                return None, None, None, None, None
            g, log_likelihood = expectation(X, pi, m, S)
            if g is None or log_likelihood is None:
                return None, None, None, None, None

        if i > 0 and abs(log_likelihood - l_prev) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(log_likelihood, 5)))
            break

        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(log_likelihood, 5)))

        l_prev = log_likelihood

    return pi, m, S, g, log_likelihood
