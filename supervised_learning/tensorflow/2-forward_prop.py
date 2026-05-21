#!/usr/bin/env python3
"""Module to create the forward propagation graph."""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x: placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer

    Returns:
        the prediction of the network in tensor form
    """
    create_layer = __import__('1-create_layer').create_layer

    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
