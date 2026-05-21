#!/usr/bin/env python3
"""Module to create the training operation."""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss: the loss of the network's prediction
        alpha: the learning rate

    Returns:
        an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
