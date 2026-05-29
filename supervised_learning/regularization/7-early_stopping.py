#!/usr/bin/env python3
"""Early stopping module."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determine if gradient descent should stop early.

    Stops when validation cost has not decreased relative to opt_cost
    by more than threshold over patience consecutive checks.

    Args:
        cost: current validation cost of the neural network.
        opt_cost: lowest recorded validation cost of the neural network.
        threshold: threshold used for early stopping.
        patience: patience count used for early stopping.
        count: current count of how long threshold has not been met.

    Returns:
        Tuple of (should_stop: bool, updated_count: int).
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count
