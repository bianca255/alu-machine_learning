#!/usr/bin/env python3
"""Create a layer with dropout module."""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Create a layer of a neural network using dropout.

    Args:
        prev: tensor containing the output of the previous layer.
        n: number of nodes the new layer should contain.
        activation: activation function to use on the layer.
        keep_prob: probability that a node will be kept.

    Returns:
        Output tensor of the new layer with dropout applied.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init)
    output = layer(prev)
    return tf.layers.dropout(output, rate=1 - keep_prob)
