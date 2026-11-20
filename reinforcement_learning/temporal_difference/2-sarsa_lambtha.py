#!/usr/bin/env python3
"""SARSA(lambda) control with epsilon-greedy exploration."""

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


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform online SARSA(lambda) with eligibility traces."""
    if (not isinstance(Q, np.ndarray) or Q.ndim != 2
            or not np.issubdtype(Q.dtype, np.number)
            or lambtha < 0 or lambtha > 1
            or episodes <= 0 or max_steps <= 0
            or alpha < 0 or gamma < 0 or gamma > 1
            or epsilon < 0 or min_epsilon < 0
            or epsilon_decay < 0):
        return None
    for _ in range(episodes):
        eligibility = np.zeros(Q.shape)
        state = _reset(env)
        if np.random.uniform() < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.random.choice(np.flatnonzero(
                Q[state] == np.max(Q[state])))
        for _ in range(max_steps):
            next_state, reward, done, _ = _step(env, action)
            if done:
                next_action = None
                target = reward
            else:
                if np.random.uniform() < epsilon:
                    next_action = np.random.randint(Q.shape[1])
                else:
                    next_action = np.random.choice(np.flatnonzero(
                        Q[next_state] == np.max(Q[next_state])))
                target = reward + gamma * Q[next_state, next_action]
            delta = target - Q[state, action]
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            state = next_state
            if done:
                break
            action = next_action
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    return Q
