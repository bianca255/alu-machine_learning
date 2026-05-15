#!/usr/bin/env python3
"""Defines a single neuron with forward propagation."""

import numpy as np


class Neuron:
    """A single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Initialize the neuron.
        
        Args:
            nx: Number of input features to the neuron
            
        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    
    @property
    def W(self):
        """Return the weights vector."""
        return self.__W
    
    @property
    def b(self):
        """Return the bias."""
        return self.__b
    
    @property
    def A(self):
        """Return the activated output."""
        return self.__A
    
    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            
        Returns:
            The private attribute __A
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A