#!/usr/bin/env python3
"""Defines a deep neural network with private attributes."""

import numpy as np


class DeepNeuralNetwork:
    """A deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.
        
        Args:
            nx: Number of input features
            layers: List representing the number of nodes in each layer
            
        Raises:
            TypeError: If nx is not an integer or layers is not a list of positive integers
            ValueError: If nx is less than 1 or layers is empty
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if not all(isinstance(ls, int) and ls > 0 for ls in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        
        prev_nodes = nx
        for i in range(1, self.__L + 1):
            self.__weights[f'W{i}'] = np.random.randn(layers[i-1], prev_nodes) * np.sqrt(2.0 / prev_nodes)
            self.__weights[f'b{i}'] = np.zeros((layers[i-1], 1))
            prev_nodes = layers[i-1]
    
    @property
    def L(self):
        """Return the number of layers."""
        return self.__L
    
    @property
    def cache(self):
        """Return the cache dictionary."""
        return self.__cache
    
    @property
    def weights(self):
        """Return the weights dictionary."""
        return self.__weights