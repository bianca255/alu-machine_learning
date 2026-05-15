#!/usr/bin/env python3
"""Defines a single neuron with upgraded training."""

import numpy as np
import matplotlib.pyplot as plt


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
        Evaluate the neuron's predictions.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) containing correct labels
            
        Returns:
            The neuron's prediction and the cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
    
    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neuron.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A: numpy.ndarray with shape (1, m) containing activated output
            alpha: Learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m
        
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
    
    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the neuron with optional verbose output and graphing.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) containing correct labels
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
            self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            
            if verbose and (i % step == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")
            
            if graph and (i % step == 0 or i == iterations):
                costs.append(cost)
                steps_list.append(i)
            
            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)
        
        if graph:
            plt.figure()
            plt.plot(steps_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        
        return self.evaluate(X, Y)
