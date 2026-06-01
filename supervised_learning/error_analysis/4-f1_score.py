#!/usr/bin/env python3
"""F1 score calculation module."""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculate the F1 score for each class in a confusion matrix.

    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    Args:
        confusion: numpy.ndarray of shape (classes, classes), confusion matrix
            where rows are correct labels and columns are predicted labels.

    Returns:
        numpy.ndarray of shape (classes,) with the F1 score of each class.
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * prec * sens / (prec + sens)
