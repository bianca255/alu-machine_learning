#!/usr/bin/env python3
"""RNN decoder with additive attention."""

import tensorflow as tf

SelfAttention = __import__("1-self_attention").SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decode one target token using encoder context."""

    def __init__(self, vocab, embedding, units, batch):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Return the next-token logits and decoder hidden state."""
        context, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, state = self.gru(x, initial_state=s_prev)
        return self.F(tf.squeeze(output, axis=1)), state
