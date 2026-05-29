#!/usr/bin/env python3
"""Gradient descent with dropout module."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update weights of a neural network with dropout using gradient descent.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels.
        weights: dictionary of weights and biases of the neural network.
        cache: dictionary of outputs and dropout masks of each layer.
        alpha: learning rate.
        keep_prob: probability that a node will be kept.
        L: number of layers of the network.

    All layers use tanh except the last which uses softmax.
    Weights are updated in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            A = cache['A' + str(i - 1)]
            D = cache['D' + str(i - 1)]
            dA = np.matmul(W.T, dZ)
            dA = dA * D / keep_prob
            dZ = dA * (1 - A ** 2)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
