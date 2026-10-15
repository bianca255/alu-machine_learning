#!/usr/bin/env python3
"""Additive attention for an RNN decoder."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculate context and alignment weights from encoder states."""

    def __init__(self, units):
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Return the weighted context vector and attention weights."""
        score = self.V(
            tf.nn.tanh(
                self.W(tf.expand_dims(s_prev, 1)) + self.U(hidden_states)
            )
        )
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
