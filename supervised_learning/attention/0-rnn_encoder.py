#!/usr/bin/env python3
"""RNN encoder for sequence-to-sequence translation."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encode token indices into a sequence of hidden states."""

    def __init__(self, vocab, embedding, units, batch):
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def initialize_hidden_state(self):
        """Return a zero-filled initial hidden state."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Encode ``x`` and return its outputs and final hidden state."""
        x = self.embedding(x)
        return self.gru(x, initial_state=initial)
