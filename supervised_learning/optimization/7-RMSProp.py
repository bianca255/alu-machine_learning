#!/usr/bin/env python3
"""RMSProp optimization module."""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha: learning rate.
        beta2: RMSProp weight.
        epsilon: small number to avoid division by zero.
        var: numpy.ndarray, variable to update.
        grad: numpy.ndarray, gradient of var.
        s: previous second moment of var.

    Returns:
        Updated variable and new second moment, respectively.
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
