#!/usr/bin/env python3
"""Variational Autoencoder module."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder (VAE).

    The encoder outputs a sampled latent vector z, the mean (mu), and the
    log variance (log_var). Sampling uses the reparameterization trick:
        z = mu + exp(log_var / 2) * epsilon,  epsilon ~ N(0, I)

    A KL-divergence loss is added to the encoder output to regularize the
    latent space toward a standard normal distribution.

    Args:
        input_dims: integer, dimensions of the model input.
        hidden_layers: list of ints, number of nodes per encoder hidden layer.
            Reversed for the decoder.
        latent_dims: integer, dimensions of the latent space.

    Returns:
        encoder: encoder model outputting (z, mu, log_var).
        decoder: decoder model.
        auto: full autoencoder model compiled with adam + binary cross-entropy.
    """
    # --- Encoder ---
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and log-variance layers (no activation)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization sampling layer
    def sampling(args):
        """Sample z using the reparameterization trick."""
        mean, lv = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
        return mean + keras.backend.exp(lv / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(encoder_input, [z, mu, log_var])

    # --- Decoder ---
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    # --- Full Autoencoder with KL loss ---
    auto_input = keras.Input(shape=(input_dims,))
    z_out, mu_out, log_var_out = encoder(auto_input)
    reconstructed = decoder(z_out)
    auto = keras.Model(auto_input, reconstructed)

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var_out
        - keras.backend.square(mu_out)
        - keras.backend.exp(log_var_out),
        axis=1
    )
    auto.add_loss(keras.backend.mean(kl_loss))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
