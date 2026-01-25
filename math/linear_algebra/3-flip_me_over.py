#!/usr/bin/env python3
"""Module for transposing a 2D matrix"""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix"""
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
