#!/usr/bin/env python3
"""Monte Carlo value-function update."""

import numpy as np


def _reset(env):
    """Support old and new Gym reset return formats."""
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _step(env, action):
    """Support old and new Gym step return formats."""
    result = env.step(action)
    if len(result) == 5:
        return result[0], result[1], result[2], result[4]
    return result[0], result[1], result[2], result[3]


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform every-visit Monte Carlo policy evaluation."""
    if (not isinstance(V, np.ndarray) or V.ndim != 1
            or not np.issubdtype(V.dtype, np.number)
            or not callable(policy) or episodes <= 0 or max_steps <= 0
            or alpha < 0 or gamma < 0 or gamma > 1):
        return None
    for _ in range(episodes):
        state = _reset(env)
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = _step(env, action)
            episode.append((state, reward))
            state = next_state
            if done:
                break
        G = 0.0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])
    return V
