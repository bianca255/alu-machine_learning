#!/usr/bin/env python3
"""Training loop for a linear softmax policy-gradient agent."""

import numpy as np
from policy_gradient import policy_gradient


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


def train(env, nb_episodes, alpha=0.000045, gamma=0.98,
          show_result=False):
    """Train CartPole and return the score from every episode."""
    if (not isinstance(nb_episodes, (int, np.integer))
            or isinstance(nb_episodes, bool) or nb_episodes <= 0
            or alpha < 0 or gamma < 0 or gamma > 1
            or not isinstance(show_result, bool)):
        return None
    n_actions = getattr(env.action_space, "n", 2)
    weight = np.random.uniform(size=(env.observation_space.shape[0],
                                     n_actions))
    scores = []
    for episode in range(nb_episodes):
        state = _reset(env)
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_gradients = []
        for _ in range(10000):
            matrix = np.asarray(state).reshape((1, -1))
            action, gradient = policy_gradient(matrix, weight)
            if action is None:
                return None
            episode_states.append(matrix)
            episode_actions.append(action)
            episode_gradients.append(gradient)
            next_state, reward, done, _ = _step(env, action)
            episode_rewards.append(reward)
            state = next_state
            if show_result and episode > 0 and episode % 1000 == 0:
                if hasattr(env, "render"):
                    env.render()
            if done:
                break
        discounted = np.zeros(len(episode_rewards))
        running = 0.0
        for i in range(len(episode_rewards) - 1, -1, -1):
            running = episode_rewards[i] + gamma * running
            discounted[i] = running
        for gradient, return_value in zip(episode_gradients, discounted):
            weight += alpha * gradient * return_value
        score = sum(episode_rewards)
        scores.append(score)
        print("Episode: {}/{}\tScore: {}".format(
            episode + 1, nb_episodes, score), end="\r", flush=False)
    return scores
