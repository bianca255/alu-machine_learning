#!/usr/bin/env python3
"""Create padding and look-ahead masks for Transformer training."""

import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Return encoder, combined decoder, and cross-attention masks."""
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]
    target_padding = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding = target_padding[:, tf.newaxis, tf.newaxis, :]
    length = tf.shape(target)[1]
    look_ahead = 1 - tf.linalg.band_part(
        tf.ones((length, length)), -1, 0
    )
    combined_mask = tf.maximum(target_padding, look_ahead)
    return encoder_mask, combined_mask, decoder_mask
