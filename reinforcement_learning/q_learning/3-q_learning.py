#!/usr/bin/env python3
"""Q-learning training."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


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


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Train a Q-learning agent and return Q plus episode rewards."""
    if (not isinstance(Q, np.ndarray) or Q.ndim != 2 or episodes <= 0
            or max_steps <= 0 or alpha < 0 or gamma < 0 or gamma > 1
            or epsilon < 0 or min_epsilon < 0 or epsilon_decay < 0):
        return None, None
    rewards = []
    for _ in range(episodes):
        state = _reset(env)
        total = 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _ = _step(env, action)
            if done and reward == 0:
                reward = -1
            Q[state, action] += alpha * (
                reward + (0 if done else gamma * np.max(Q[next_state]))
                - Q[state, action])
            total += reward
            state = next_state
            if done:
                break
        rewards.append(total)
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    return Q, rewards
