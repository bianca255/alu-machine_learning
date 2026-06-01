#!/usr/bin/env python3
"""Sensitivity (recall) calculation module."""
import numpy as np


def sensitivity(confusion):
    """Calculate the sensitivity for each class in a confusion matrix.

    Sensitivity (recall) = TP / (TP + FN) = correct predictions per class
    divided by the total actual instances of that class (row sum).

    Args:
        confusion: numpy.ndarray of shape (classes, classes), confusion matrix
            where rows are correct labels and columns are predicted labels.

    Returns:
        numpy.ndarray of shape (classes,) with the sensitivity of each class.
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
