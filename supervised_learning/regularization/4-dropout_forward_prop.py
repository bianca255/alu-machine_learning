#!/usr/bin/env python3
"""Forward propagation with dropout module."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conduct forward propagation using dropout.

    Args:
        X: numpy.ndarray of shape (nx, m) containing input data.
        weights: dictionary of weights and biases of the neural network.
        L: number of layers in the network.
        keep_prob: probability that a node will be kept.

    Returns:
        Dictionary containing the outputs of each layer and the dropout
        masks used on each layer (keys A0..AL and D1..D(L-1)).
    """
    cache = {'A0': X}

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Softmax for the last layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache['A' + str(i)] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Tanh + dropout for hidden layers
            A = np.tanh(Z)
            D = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A = A * D / keep_prob
            cache['A' + str(i)] = A
            cache['D' + str(i)] = D

    return cache
