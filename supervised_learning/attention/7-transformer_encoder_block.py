#!/usr/bin/env python3
"""Transformer encoder block."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Apply attention and a feed-forward network with residual paths."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Return the transformed encoder sequence."""
        attention, _ = self.mha(x, x, x, mask=mask)
        attention = self.dropout1(attention, training=training)
        output = self.layernorm1(x + attention)
        feed_forward = self.dense_output(self.dense_hidden(output))
        feed_forward = self.dropout2(feed_forward, training=training)
        return self.layernorm2(output + feed_forward)
