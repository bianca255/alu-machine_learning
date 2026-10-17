#!/usr/bin/env python3
"""Transformer model used by the translation training loop."""

import tensorflow.compat.v2 as tf


def scaled_attention(query, key, value, mask=None):
    """Calculate scaled dot-product attention."""
    logits = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = logits / tf.math.sqrt(depth)
    if mask is not None:
        logits += mask * -1e9
    weights = tf.nn.softmax(logits, axis=-1)
    return tf.matmul(weights, value), weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Project inputs into multiple scaled dot-product attention heads."""

    def __init__(self, dm, h):
        """Create the projections used by the attention heads."""
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x):
        """Split a projected sequence into attention heads."""
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1], self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None):
        """Return combined attention output and attention weights."""
        query = self.split_heads(self.Wq(query))
        key = self.split_heads(self.Wk(key))
        value = self.split_heads(self.Wv(value))
        output, weights = scaled_attention(query, key, value, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        shape = tf.shape(output)
        output = tf.reshape(output, (shape[0], -1, self.dm))
        return self.linear(output), weights


class EncoderBlock(tf.keras.layers.Layer):
    """Apply self-attention and a feed-forward residual block."""

    def __init__(self, dm, h, hidden, drop_rate):
        """Create the encoder block layers."""
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Return the encoded sequence for this block."""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        output = self.layernorm1(x + attention)
        feed_forward = self.dense_output(self.dense_hidden(output))
        feed_forward = self.dropout2(feed_forward, training=training)
        return self.layernorm2(output + feed_forward)


class DecoderBlock(tf.keras.layers.Layer):
    """Apply masked attention and a feed-forward residual block."""

    def __init__(self, dm, h, hidden, drop_rate):
        """Create the decoder block layers."""
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

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Return the decoded sequence for this block."""
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        output1 = self.layernorm1(x + attention1)
        attention2, _ = self.mha2(
            output1, encoder_output, encoder_output, padding_mask
        )
        attention2 = self.dropout2(attention2, training=training)
        output2 = self.layernorm2(output1 + attention2)
        feed_forward = self.dense_output(self.dense_hidden(output2))
        feed_forward = self.dropout3(feed_forward, training=training)
        return self.layernorm3(output2 + feed_forward)


class Encoder(tf.keras.layers.Layer):
    """Embed input tokens and pass them through encoder blocks."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate):
        """Create the embedding, positional encoding, and encoder blocks."""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Return the encoded input sequence."""
        length = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:length]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """Embed target tokens and pass them through decoder blocks."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate):
        """Create the embedding, positional encoding, and decoder blocks."""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Return the decoded target sequence."""
        length = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:length]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(
                x, encoder_output, training,
                look_ahead_mask, padding_mask,
            )
        return x


def positional_encoding(max_seq_len, dm):
    """Return sinusoidal positional encodings without external imports."""
    positions = tf.cast(tf.range(max_seq_len)[:, tf.newaxis], tf.float32)
    dimensions = tf.cast(tf.range(dm)[tf.newaxis, :], tf.float32)
    angles = positions / tf.pow(
        10000.0,
        (2 * tf.floor(dimensions / 2)) / tf.cast(dm, tf.float32),
    )
    even = tf.sin(angles[:, 0::2])
    odd = tf.cos(angles[:, 1::2])
    return tf.reshape(tf.stack([even, odd], axis=-1), (max_seq_len, dm))


class Transformer(tf.keras.Model):
    """Combine an encoder, decoder, and target vocabulary projection."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Create the encoder, decoder, and output projection."""
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Return target-vocabulary logits for input and target tokens."""
        encoded = self.encoder(inputs, training=training, mask=encoder_mask)
        decoded = self.decoder(
            target,
            encoded,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask,
        )
        return self.linear(decoded)
