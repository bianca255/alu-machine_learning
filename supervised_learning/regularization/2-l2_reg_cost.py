#!/usr/bin/env python3
"""L2 regularization cost module (TensorFlow)."""
import tensorflow as tf


def l2_reg_cost(cost):
    """Calculate the cost of a neural network with L2 regularization.

    Args:
        cost: tensor containing the cost without L2 regularization.

    Returns:
        Tensor containing the cost accounting for L2 regularization.
    """
    return cost + tf.losses.get_regularization_losses()
