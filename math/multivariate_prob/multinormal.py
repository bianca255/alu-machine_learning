#!/usr/bin/env python3
"""Multivariate Normal distribution class"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        Initialize MultiNormal distribution

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
                  n is the number of data points
                  d is the number of dimensions in each data point

        Raises:
            TypeError: If data is not a 2D numpy.ndarray
            ValueError: If n is less than 2
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean - shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate covariance matrix - shape (d, d)
        # Cov = (1/(n-1)) * (data - mean) * (data - mean)^T
        # Using sample covariance (Bessel's correction)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculate the PDF at a data point

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point

        Returns:
            The value of the PDF at x

        Raises:
            TypeError: If x is not a numpy.ndarray
            ValueError: If x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # PDF formula for multivariate normal:
        # f(x) = (1 / sqrt((2π)^d * |Σ|)) * exp(-0.5 * (x-μ)^T * Σ^-1 * (x-μ))

        # Calculate determinant of covariance matrix
        det_cov = np.linalg.det(self.cov)

        # Calculate inverse of covariance matrix
        inv_cov = np.linalg.inv(self.cov)

        # Calculate (x - mean)
        x_centered = x - self.mean

        # Calculate the exponent: -0.5 * (x-μ)^T * Σ^-1 * (x-μ)
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv_cov), x_centered)

        # Calculate the coefficient: 1 / sqrt((2π)^d * |Σ|)
        coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det_cov)

        # Calculate PDF
        pdf_value = coefficient * np.exp(exponent)

        # Return scalar value
        return pdf_value[0][0]
