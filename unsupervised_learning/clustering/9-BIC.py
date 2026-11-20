#!/usr/bin/env python3
"""Find the best number of clusters for a GMM using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using BIC.

    X: numpy.ndarray of shape (n, d) containing the data set
    kmin: positive integer, minimum number of clusters (inclusive)
    kmax: positive integer, maximum number of clusters (inclusive)
    iterations: positive integer, maximum EM iterations
    tol: non-negative float, tolerance for EM
    verbose: boolean, whether EM prints information

    Returns: best_k, best_result, l, b, or None, None, None, None
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0 or kmin > n:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmax > n:
        return None, None, None, None
    if kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    l_list = []
    b_list = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None:
            return None, None, None, None

        p = (k - 1) + k * d + k * d * (d + 1) / 2
        bic = p * np.log(n) - 2 * ll

        l_list.append(ll)
        b_list.append(bic)
        results.append((pi, m, S))

    log_likelihoods = np.array(l_list)
    b = np.array(b_list)
    best = np.argmin(b)
    best_k = best + kmin
    best_result = results[best]
    return best_k, best_result, log_likelihoods, b
