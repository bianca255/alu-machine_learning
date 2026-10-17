#!/usr/bin/env python3
"""Perform forward propagation for a simple RNN."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation over a sequence.

    Args:
        rnn_cell: RNNCell instance used for propagation.
        X: Input data of shape (t, m, i).
        h_0: Initial hidden state of shape (m, h).

    Returns:
        The hidden states, including the initial state, and outputs.
    """
    time_steps = X.shape[0]
    hidden_size = h_0.shape[1]
    outputs = []
    hidden_states = np.zeros((time_steps + 1, X.shape[1], hidden_size))
    hidden_states[0] = h_0
    hidden = h_0
    for time in range(time_steps):
        hidden, output = rnn_cell.forward(hidden, X[time])
        hidden_states[time + 1] = hidden
        outputs.append(output)
    return hidden_states, np.array(outputs)
