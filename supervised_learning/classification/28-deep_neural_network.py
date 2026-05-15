#!/usr/bin/env python3
"""Defines a deep neural network with selectable activation functions."""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """A deep neural network performing multiclass classification with selectable activations."""

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize the deep neural network.
        
        Args:
            nx: Number of input features
            layers: List representing the number of nodes in each layer
            activation: Activation function ('sig' for sigmoid, 'tanh' for tanh)
            
        Raises:
            TypeError: If nx is not an integer or layers is not a list of positive integers
            ValueError: If nx is less than 1 or layers is empty or activation is invalid
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
        
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        
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
    
    @property
    def activation(self):
        """Return the activation function."""
        return self.__activation
    
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
            if i == self.__L:
                # Use softmax for output layer in multiclass
                A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
            else:
                # Use selected activation for hidden layers
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:  # tanh
                    A = np.tanh(Z)
            self.__cache[f'A{i}'] = A
        
        return A, self.__cache
    
    def cost(self, Y, A):
        """
        Calculate the cost of the model using cross-entropy loss.
        
        Args:
            Y: numpy.ndarray with shape (classes, m) containing correct labels (one-hot)
            A: numpy.ndarray with shape (classes, m) containing activated output
            
        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-10)) / m
        return cost
    
    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (classes, m) containing correct labels (one-hot)
            
        Returns:
            The neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.eye(Y.shape[0])[np.argmax(A, axis=0)].T
        return prediction, cost
    
    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neural network.
        
        Args:
            Y: numpy.ndarray containing correct labels
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
                if self.__activation == 'sig':
                    dZ = dA * A_prev * (1 - A_prev)
                else:  # tanh
                    dZ = dA * (1 - A_prev**2)
            
            self.__weights[f'W{i}'] = self.__weights[f'W{i}'] - alpha * dW
            self.__weights[f'b{i}'] = self.__weights[f'b{i}'] - alpha * db
    
    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the deep neural network with optional verbose output and graphing.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray containing correct labels
            iterations: Number of iterations to train over
            alpha: Learning rate
            verbose: Boolean to print information about training
            graph: Boolean to graph information about training
            step: Step interval for output
            
        Returns:
            The evaluation of the training data after iterations
            
        Raises:
            TypeError: If iterations is not an integer, alpha is not a float,
                       or step is not an integer (when verbose or graph is True)
            ValueError: If iterations or alpha is not positive, or step is
                        not positive or greater than iterations
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        
        costs = []
        steps_list = []
        
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            
            if verbose and (i % step == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")
            
            if graph and (i % step == 0 or i == iterations):
                costs.append(cost)
                steps_list.append(i)
            
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
        
        if graph:
            plt.figure()
            plt.plot(steps_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        
        return self.evaluate(X, Y)
    
    def save(self, filename):
        """
        Save the instance object to a file in pickle format.
        
        Args:
            filename: File to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        """
        Load a pickled DeepNeuralNetwork object.
        
        Args:
            filename: File from which the object should be loaded
            
        Returns:
            The loaded object, or None if filename doesn't exist
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
