#!/usr/bin/env python3
"""Transformer encoder-decoder network."""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Combine an encoder, decoder, and target vocabulary projection."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Return target-vocabulary logits for the input and target tokens."""
        encoder_output = self.encoder(inputs, training=training,
                                      mask=encoder_mask)
        decoder_output = self.decoder(
            target,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask,
        )
        return self.linear(decoder_output)
