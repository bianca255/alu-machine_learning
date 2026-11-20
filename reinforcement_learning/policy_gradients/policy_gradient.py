#!/usr/bin/env python3
"""Policy-gradient functions for a linear softmax policy."""

import numpy as np


def policy(matrix, weight):
    """Compute a stable softmax probability distribution."""
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2
            or not isinstance(weight, np.ndarray) or weight.ndim != 2
            or matrix.shape[1] != weight.shape[0]):
        return None
    logits = np.matmul(matrix, weight)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Sample an action and calculate its policy-gradient weight update."""
    probabilities = policy(state, weight)
    if probabilities is None:
        return None, None
    action = np.random.choice(probabilities.shape[1],
                              p=probabilities[0])
    one_hot = np.zeros_like(probabilities)
    one_hot[0, action] = 1
    gradient = np.matmul(state.T, one_hot - probabilities)
    return action, gradient
