#!/usr/bin/env python3
"""Module for calculating minor matrix"""


def determinant(matrix):
    """Calculate the determinant of a matrix"""
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix)):
        submatrix = [[matrix[i][k] for k in range(len(matrix)) if k != j]
                     for i in range(1, len(matrix))]
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(submatrix)

    return det


def minor(matrix):
    """Calculate the minor matrix of a matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = [[matrix[r][c] for c in range(n) if c != j]
                         for r in range(n) if r != i]
            minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix
