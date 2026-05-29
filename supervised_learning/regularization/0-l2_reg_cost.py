#!/usr/bin/env python3
"""L2 regularization cost module."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculate the cost of a neural network with L2 regularization.

    Args:
        cost: cost of the network without L2 regularization.
        lambtha: regularization parameter.
        weights: dictionary of weights and biases (numpy.ndarrays).
        L: number of layers in the neural network.
        m: number of data points used.

    Returns:
        Cost of the network accounting for L2 regularization.
    """
    frobenius = sum(
        np.linalg.norm(weights['W' + str(i)]) ** 2
        for i in range(1, L + 1)
    )
    return cost + (lambtha / (2 * m)) * frobenius
