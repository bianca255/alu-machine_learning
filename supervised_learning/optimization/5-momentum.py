#!/usr/bin/env python3
"""Gradient descent with momentum module."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using gradient descent with momentum.

    Args:
        alpha: learning rate.
        beta1: momentum weight.
        var: numpy.ndarray, variable to update.
        grad: numpy.ndarray, gradient of var.
        v: previous first moment of var.

    Returns:
        Updated variable and new moment, respectively.
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
