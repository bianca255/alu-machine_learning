#!/usr/bin/env python3
"""Specificity calculation module."""
import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in a confusion matrix.

    Specificity = TN / (TN + FP). For each class, TN is the sum of all
    entries that are neither in that class's row nor its column, FP is
    the column sum minus the diagonal, and FN is the row sum minus the
    diagonal.

    Args:
        confusion: numpy.ndarray of shape (classes, classes), confusion matrix
            where rows are correct labels and columns are predicted labels.

    Returns:
        numpy.ndarray of shape (classes,) with the specificity of each class.
    """
    total = np.sum(confusion)
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = total - tp - fp - fn
    return tn / (tn + fp)
