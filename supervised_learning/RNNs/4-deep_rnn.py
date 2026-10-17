#!/usr/bin/env python3
"""Perform forward propagation for a deep RNN."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation through multiple recurrent layers."""
    time_steps, batch_size = X.shape[:2]
    layers = len(rnn_cells)
    hidden_size = h_0.shape[2]
    hidden_states = np.zeros((time_steps + 1, layers, batch_size, hidden_size))
    hidden_states[0] = h_0
    outputs = []
    for time in range(time_steps):
        layer_input = X[time]
        for layer, cell in enumerate(rnn_cells):
            hidden, output = cell.forward(hidden_states[time, layer],
                                          layer_input)
            hidden_states[time + 1, layer] = hidden
            layer_input = hidden
            if layer == layers - 1:
                outputs.append(output)
    return hidden_states, np.array(outputs)
