#!/usr/bin/env python3
"""Gradient descent with L2 regularization module."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update weights using gradient descent with L2 regularization.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels.
        weights: dictionary of weights and biases of the neural network.
        cache: dictionary of outputs of each layer.
        alpha: learning rate.
        lambtha: L2 regularization parameter.
        L: number of layers of the network.

    The network uses tanh activations on all layers except the last,
    which uses softmax. Weights and biases are updated in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        dW = np.matmul(dZ, A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            A = cache['A' + str(i - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - A ** 2)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
