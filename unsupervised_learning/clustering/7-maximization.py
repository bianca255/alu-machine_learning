#!/usr/bin/env python3
"""Maximization step of the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a GMM.

    X: numpy.ndarray of shape (n, d) containing the data set
    g: numpy.ndarray of shape (k, n) containing the posterior
       probabilities for each data point in each cluster

    Returns: pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) updated priors
        m: numpy.ndarray of shape (k, d) updated centroid means
        S: numpy.ndarray of shape (k, d, d) updated covariance matrices
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None

    N_k = np.sum(g, axis=1)
    pi = N_k / n
    m = np.matmul(g, X) / N_k[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted = g[i][:, np.newaxis] * diff
        S[i] = np.matmul(weighted.T, diff) / N_k[i]
    return pi, m, S
