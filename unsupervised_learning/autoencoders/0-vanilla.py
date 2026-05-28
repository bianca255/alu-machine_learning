#!/usr/bin/env python3
"""Vanilla Autoencoder module."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a vanilla autoencoder.

    Args:
        input_dims: integer, dimensions of the model input.
        hidden_layers: list of ints, number of nodes per encoder hidden layer.
            Reversed for the decoder.
        latent_dims: integer, dimensions of the latent space.

    Returns:
        encoder: the encoder model.
        decoder: the decoder model.
        auto: the full autoencoder model.
    """
    # --- Encoder ---
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(encoder_input, latent)

    # --- Decoder ---
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    # --- Full Autoencoder ---
    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
