#!/usr/bin/env python3
"""Module for calculating matrix determinant"""


def determinant(matrix):
    """Calculate the determinant of a matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        submatrix = [[matrix[i][k] for k in range(n) if k != j]
                     for i in range(1, n)]
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(submatrix)

    return det
