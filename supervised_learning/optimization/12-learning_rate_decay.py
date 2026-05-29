#!/usr/bin/env python3
"""Learning rate decay module (TensorFlow)."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Create a stepwise inverse time decay learning rate operation.

    Args:
        alpha: original learning rate.
        decay_rate: weight determining the rate of decay.
        global_step: number of gradient descent passes elapsed.
        decay_step: passes before alpha is decayed again.

    Returns:
        Learning rate decay operation.
    """
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
