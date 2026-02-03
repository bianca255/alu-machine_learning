#!/usr/bin/env python3
"""
Calculate the sum of squares from 1 to n: sum(i^2) for i in 1..n
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from i=1 to n
    
    Args:
        n: The stopping condition (upper limit of summation)
    
    Returns:
        The integer value of the sum, or None if n is invalid
    """
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return None
    
    if n <= 0:
        return None
    
    # Using the formula: sum(i^2) = n(n+1)(2n+1)/6
    n = int(n)
    return n * (n + 1) * (2 * n + 1) // 6
