#!/usr/bin/env python3
"""Moving average module."""


def moving_average(data, beta):
    """Calculate the weighted moving average with bias correction.

    Args:
        data: list of data points.
        beta: weight for the moving average.

    Returns:
        List of moving averages.
    """
    averages = []
    v = 0
    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        averages.append(v / (1 - beta ** t))
    return averages
