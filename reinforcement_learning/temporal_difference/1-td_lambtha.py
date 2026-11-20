#!/usr/bin/env python3
"""TD(lambda) value-function update."""

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


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Perform online TD(lambda) with accumulating eligibility traces."""
    if (not isinstance(V, np.ndarray) or V.ndim != 1
            or not np.issubdtype(V.dtype, np.number)
            or not callable(policy) or lambtha < 0 or lambtha > 1
            or episodes <= 0 or max_steps <= 0
            or alpha < 0 or gamma < 0 or gamma > 1):
        return None
    for _ in range(episodes):
        eligibility = np.zeros(V.shape)
        state = _reset(env)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = _step(env, action)
            target = reward
            if not done:
                target += gamma * V[next_state]
            delta = target - V[state]
            eligibility[state] += 1
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            state = next_state
            if done:
                break
    return V
