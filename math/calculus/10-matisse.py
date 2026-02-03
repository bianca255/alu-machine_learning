#!/usr/bin/env python3
"""
Calculate the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial

    Args:
        poly: A list of coefficients where index represents the power of x
              Example: [5, 3, 0, 1] represents 5 + 3x + 0x^2 + 1x^3

    Returns:
        A new list of coefficients representing the derivative,
        or None if invalid
    """
    if poly is None or not isinstance(poly, list):
        return None

    if len(poly) == 0:
        return None

    # Check if all elements are numbers
    for coeff in poly:
        if not isinstance(coeff, (int, float)) or isinstance(coeff, bool):
            return None

    # If polynomial is constant, derivative is 0
    if len(poly) == 1:
        return [0]

    # Calculate derivative: d/dx of c_i * x^i is i * c_i * x^(i-1)
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    return derivative
