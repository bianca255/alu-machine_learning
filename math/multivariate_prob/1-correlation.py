#!/usr/bin/env python3
"""Calculate correlation matrix from covariance matrix"""
import numpy as np


def correlation(C):
    """
    Calculate a correlation matrix from a covariance matrix

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix
           d is the number of dimensions

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix

    Raises:
        TypeError: If C is not a numpy.ndarray
        ValueError: If C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Get the diagonal elements (variances)
    variances = np.diag(C)

    # Calculate standard deviations
    std_devs = np.sqrt(variances)

    # Create outer product of standard deviations
    std_matrix = np.outer(std_devs, std_devs)

    # Calculate correlation matrix
    correlation_matrix = C / std_matrix

    return correlation_matrix
