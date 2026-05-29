#!/usr/bin/env python3
"""Create a layer with L2 regularization module."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create a tensorflow layer with L2 regularization.

    Args:
        prev: tensor containing the output of the previous layer.
        n: number of nodes the new layer should contain.
        activation: activation function to use on the layer.
        lambtha: L2 regularization parameter.

    Returns:
        Output tensor of the new layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=regularizer
    )
    return layer(prev)
