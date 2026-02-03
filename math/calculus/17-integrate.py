#!/usr/bin/env python3
"""
Calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial
    
    Args:
        poly: A list of coefficients where index represents the power of x
              Example: [5, 3, 0, 1] represents 5 + 3x + 0x^2 + 1x^3
        C: The integration constant (default 0)
    
    Returns:
        A new list of coefficients representing the integral, or None if invalid
    """
    if poly is None or not isinstance(poly, list):
        return None
    
    if len(poly) == 0:
        return None
    
    # Check if all elements are numbers
    for coeff in poly:
        if not isinstance(coeff, (int, float)) or isinstance(coeff, bool):
            return None
    
    # Check if C is valid
    if not isinstance(C, (int, float)) or isinstance(C, bool):
        return None
    
    # Calculate integral: integral of c_i * x^i is (c_i / (i+1)) * x^(i+1)
    integral = [C]
    
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        # Convert to int if it's a whole number
        if isinstance(coeff, float) and coeff.is_integer():
            coeff = int(coeff)
        integral.append(coeff)
    
    # Remove trailing zeros, but keep at least [C] or [0]
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    
    return integral
