#!/usr/bin/env python3
"""TensorFlow batch normalization layer module."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer for a neural network.

    Args:
        prev: activated output of the previous layer.
        n: number of nodes in the layer to be created.
        activation: activation function for the layer output.

    Returns:
        Tensor of the activated output for the layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(n, kernel_initializer=init)
    Z = dense(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-8)

    if activation is None:
        return Z_norm
    return activation(Z_norm)
