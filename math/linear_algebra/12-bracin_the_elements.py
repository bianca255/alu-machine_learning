#!/usr/bin/env python3
"""Module for element-wise operations on numpy arrays"""


def np_elementwise(mat1, mat2):
    """Perform element-wise operations on two matrices"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
