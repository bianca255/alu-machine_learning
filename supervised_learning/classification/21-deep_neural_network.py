#!/usr/bin/env python3
"""Defines a deep neural network with gradient descent."""

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
        
        for layer_size in layers:
            if not isinstance(layer_size, int) or layer_size < 1:
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
    
    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            
        Returns:
            The output of the neural network and the cache
        """
        self.__cache['A0'] = X
        A = X
        
        for i in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f'W{i}'], A) + self.__weights[f'b{i}']
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{i}'] = A
        
        return A, self.__cache
    
    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression.
        
        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A: numpy.ndarray with shape (1, m) containing activated output
            
        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
    
    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) containing correct labels
            
        Returns:
            The neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
    
    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neural network.
        
        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            cache: Dictionary containing all the intermediary values
            alpha: Learning rate
        """
        m = Y.shape[1]
        dZ = cache[f'A{self.__L}'] - Y
        
        for i in range(self.__L, 0, -1):
            A_prev = cache[f'A{i-1}']
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            if i > 1:
                dA = np.matmul(self.__weights[f'W{i}'].T, dZ)
                dZ = dA * A_prev * (1 - A_prev)
            
            self.__weights[f'W{i}'] = self.__weights[f'W{i}'] - alpha * dW
            self.__weights[f'b{i}'] = self.__weights[f'b{i}'] - alpha * db