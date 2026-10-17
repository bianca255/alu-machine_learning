#!/usr/bin/env python3
"""Scaled dot-product attention."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Return attention output and weights for query, key, and value."""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    scale = tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
    scaled_attention_logits = matmul_qk / scale
    if mask is not None:
        scaled_attention_logits += mask * -1e9
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
