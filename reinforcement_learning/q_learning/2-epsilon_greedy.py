#!/usr/bin/env python3
"""Epsilon-greedy action selection."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose a random action with probability epsilon."""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, Q.shape[1])
    return np.argmax(Q[state, :])
