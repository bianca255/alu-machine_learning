#!/usr/bin/env python3
"""Confusion matrix creation module."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix from one-hot labels and predictions.

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes), correct labels.
        logits: one-hot numpy.ndarray of shape (m, classes), predicted labels.

    Returns:
        Confusion matrix of shape (classes, classes) where rows are correct
        labels and columns are predicted labels.
    """
    return np.matmul(labels.T, logits)
