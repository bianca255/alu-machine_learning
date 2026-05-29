#!/usr/bin/env python3
"""TensorFlow RMSProp optimization module."""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Create an RMSProp training operation.

    Args:
        loss: loss tensor of the network.
        alpha: learning rate.
        beta2: RMSProp weight.
        epsilon: small number to avoid division by zero.

    Returns:
        RMSProp optimization operation.
    """
    return tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                     epsilon=epsilon).minimize(loss)
