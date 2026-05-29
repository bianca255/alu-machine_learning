#!/usr/bin/env python3
"""Adam optimization module."""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable using the Adam optimization algorithm.

    Args:
        alpha: learning rate.
        beta1: weight for the first moment.
        beta2: weight for the second moment.
        epsilon: small number to avoid division by zero.
        var: numpy.ndarray, variable to update.
        grad: numpy.ndarray, gradient of var.
        v: previous first moment of var.
        s: previous second moment of var.
        t: time step for bias correction.

    Returns:
        Updated variable, new first moment, new second moment, respectively.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2

    v_corr = v / (1 - beta1 ** t)
    s_corr = s / (1 - beta2 ** t)

    var = var - alpha * v_corr / (np.sqrt(s_corr) + epsilon)
    return var, v, s
