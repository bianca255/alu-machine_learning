#!/usr/bin/env python3
"""Precision calculation module."""
import numpy as np


def precision(confusion):
    """Calculate the precision for each class in a confusion matrix.

    Precision = TP / (TP + FP) = correct predictions per class divided by
    the total predicted instances of that class (column sum).

    Args:
        confusion: numpy.ndarray of shape (classes, classes), confusion matrix
            where rows are correct labels and columns are predicted labels.

    Returns:
        numpy.ndarray of shape (classes,) with the precision of each class.
    """
    return np.diag(confusion) / np.sum(confusion, axis=0)
