#!/usr/bin/env python3
"""Define a gated recurrent unit cell."""

import numpy as np


class GRUCell:
    """Represent a gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(value):
        """Compute a numerically stable sigmoid."""
        return 1 / (1 + np.exp(-np.clip(value, -500, 500)))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        combined = np.concatenate((h_prev, x_t), axis=1)
        z = self._sigmoid(np.matmul(combined, self.Wz) + self.bz)
        r = self._sigmoid(np.matmul(combined, self.Wr) + self.br)
        reset_combined = np.concatenate((r * h_prev, x_t), axis=1)
        candidate = np.tanh(np.matmul(reset_combined, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * candidate
        logits = np.matmul(h_next, self.Wy) + self.by
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return h_next, y
