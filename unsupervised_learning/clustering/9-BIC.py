#!/usr/bin/env python3
"""Bayesian Information Criterion for Gaussian mixture models."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best GMM cluster count using BIC."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0
            or not isinstance(kmin, int) or kmin <= 0
            or not isinstance(iterations, int) or iterations <= 0
            or not isinstance(tol, (int, float)) or tol < 0
            or not isinstance(verbose, bool)):
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if (not isinstance(kmax, int) or kmax < kmin or kmax > X.shape[0]):
        return None, None, None, None
    likelihoods = []
    bics = []
    results = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        result = expectation_maximization(X, k, iterations, tol, verbose)
        if result[0] is None:
            return None, None, None, None
        results.append(result)
        likelihoods.append(result[-1])
        p = k * (d + d * (d + 1) / 2 + 1) - 1
        bics.append(p * np.log(n) - 2 * result[-1])
    best = int(np.argmin(bics))
    return kmin + best, results[best][:3], np.array(likelihoods), np.array(bics)
