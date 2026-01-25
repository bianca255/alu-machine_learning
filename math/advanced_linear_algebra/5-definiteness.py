#!/usr/bin/env python3
"""Module for determining matrix definiteness"""
import numpy as np


def definiteness(matrix):
    """Calculate the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    
    if len(matrix.shape) != 2:
        return None
    
    n, m = matrix.shape
    
    if n != m or n == 0:
        return None
    
    if not np.allclose(matrix, matrix.T):
        return None
    
    eigenvalues = np.linalg.eigvals(matrix)
    
    positive = np.all(eigenvalues > 0)
    negative = np.all(eigenvalues < 0)
    positive_semi = np.all(eigenvalues >= 0) and np.any(eigenvalues == 0)
    negative_semi = np.all(eigenvalues <= 0) and np.any(eigenvalues == 0)
    
    if positive:
        return "Positive definite"
    elif positive_semi:
        return "Positive semi-definite"
    elif negative:
        return "Negative definite"
    elif negative_semi:
        return "Negative semi-definite"
    else:
        return "Indefinite"
