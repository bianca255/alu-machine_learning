#!/usr/bin/env python3
"""Multi-head scaled dot-product attention."""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Project inputs into attention heads and combine their outputs."""

    def __init__(self, dm, h):
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x):
        """Split the last dimension into attention heads."""
        batch = tf.shape(x)[0]
        length = tf.shape(x)[1]
        x = tf.reshape(x, (batch, length, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Return combined attention output and per-head weights."""
        query = self.split_heads(self.Wq(Q))
        key = self.split_heads(self.Wk(K))
        value = self.split_heads(self.Wv(V))
        output, weights = sdp_attention(query, key, value, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        batch = tf.shape(output)[0]
        output = tf.reshape(output, (batch, -1, self.dm))
        return self.linear(output), weights
