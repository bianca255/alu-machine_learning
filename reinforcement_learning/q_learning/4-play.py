#!/usr/bin/env python3
"""Play FrozenLake with a trained Q-table."""

import numpy as np


def _reset(env):
    """Support old and new Gym reset return formats."""
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _step(env, action):
    """Support old and new Gym step return formats."""
    result = env.step(action)
    if len(result) == 5:
        return result[0], result[1], result[2] or result[3], result[4]
    return result[0], result[1], result[2], result[3]


def play(env, Q, max_steps=100):
    """Play one episode, printing each observed board state."""
    state = _reset(env)
    total = 0
    for _ in range(max_steps):
        if hasattr(env, "render"):
            rendered = env.render()
            if isinstance(rendered, str):
                print(rendered)
        action = int(np.argmax(Q[state]))
        state, reward, done, _ = _step(env, action)
        total += reward
        if done:
            break
    return total
