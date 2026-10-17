#!/usr/bin/env python3
"""Perform forward propagation for a bidirectional RNN."""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation in both directions."""
    time_steps, batch_size = X.shape[:2]
    hidden_size = h_0.shape[1]
    forward_states = np.zeros((time_steps + 1, batch_size, hidden_size))
    backward_states = np.zeros((time_steps + 1, batch_size, hidden_size))
    forward_states[0] = h_0
    backward_states[time_steps] = h_t
    for time in range(time_steps):
        forward_states[time + 1] = bi_cell.forward(forward_states[time],
                                                   X[time])
    for time in range(time_steps - 1, -1, -1):
        backward_states[time] = bi_cell.backward(backward_states[time + 1],
                                                 X[time])
    hidden = np.concatenate((forward_states[1:], backward_states[:-1]),
                            axis=2)
    return hidden, bi_cell.output(hidden)
