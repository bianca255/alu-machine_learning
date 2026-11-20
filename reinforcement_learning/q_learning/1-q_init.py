#!/usr/bin/env python3
"""Initialize a Q-table."""

import numpy as np


def q_init(env):
    """Return a zero-filled state/action Q-table."""
    return np.zeros((env.observation_space.n, env.action_space.n))
