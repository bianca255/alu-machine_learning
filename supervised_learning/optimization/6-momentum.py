#!/usr/bin/env python3
"""TensorFlow momentum optimization module."""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Create a momentum gradient descent training operation.

    Args:
        loss: loss tensor of the network.
        alpha: learning rate.
        beta1: momentum weight.

    Returns:
        Momentum optimization operation.
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
