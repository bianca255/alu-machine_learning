#!/usr/bin/env python3
"""Define the backward half of a bidirectional RNN cell."""

import numpy as np


class BidirectionalCell:
    """Represent a bidirectional RNN cell with backward propagation."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases."""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculate the next hidden state in the forward direction."""
        combined = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(np.matmul(combined, self.Whf) + self.bhf)

    def backward(self, h_next, x_t):
        """Calculate the previous hidden state backward in time."""
        combined = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(np.matmul(combined, self.Whb) + self.bhb)
