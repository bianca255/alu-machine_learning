#!/usr/bin/env python3
"""Transformer decoder block."""

import tensorflow as tf

MultiHeadAttention = __import__("6-multihead_attention").MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Apply masked attention and a feed-forward network."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Return the transformed decoder sequence."""
        attention1, _ = self.mha1(x, x, x, mask=look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        output1 = self.layernorm1(x + attention1)
        attention2, _ = self.mha2(
            output1, encoder_output, encoder_output, mask=padding_mask
        )
        attention2 = self.dropout2(attention2, training=training)
        output2 = self.layernorm2(output1 + attention2)
        feed_forward = self.dense_output(self.dense_hidden(output2))
        feed_forward = self.dropout3(feed_forward, training=training)
        return self.layernorm3(output2 + feed_forward)
