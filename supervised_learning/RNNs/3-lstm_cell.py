#!/usr/bin/env python3
"""Define a long short-term memory cell."""

import numpy as np


class LSTMCell:
    """Represent a long short-term memory unit."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases."""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(value):
        """Compute a numerically stable sigmoid."""
        return 1 / (1 + np.exp(-np.clip(value, -500, 500)))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step."""
        combined = np.concatenate((h_prev, x_t), axis=1)
        forget = self._sigmoid(np.matmul(combined, self.Wf) + self.bf)
        update = self._sigmoid(np.matmul(combined, self.Wu) + self.bu)
        candidate = np.tanh(np.matmul(combined, self.Wc) + self.bc)
        c_next = forget * c_prev + update * candidate
        output = self._sigmoid(np.matmul(combined, self.Wo) + self.bo)
        h_next = output * np.tanh(c_next)
        logits = np.matmul(h_next, self.Wy) + self.by
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return h_next, c_next, y
