#!/usr/bin/env python3
"""Defines a neural network with upgraded training."""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """A neural network with one hidden layer performing binary classification."""

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        Args:
            nx: Number of input features
            nodes: Number of nodes found in the hidden layer

        Raises:
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Return the weights vector for the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Return the bias for the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Return the activated output for the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Return the weights vector for the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Return the bias for the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Return the activated output for the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network.

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data

        Returns:
            The private attributes __A1 and __A2, respectively
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

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
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neural network.

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A1: Output of the hidden layer
            A2: Predicted output
            alpha: Learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.matmul(self.__W2.T, dZ2)
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the neural network with optional verbose output and graphing.

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
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)

            if verbose and (i % step == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")

            if graph and (i % step == 0 or i == iterations):
                costs.append(cost)
                steps_list.append(i)

            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)

        if graph:
            plt.figure()
            plt.plot(steps_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
