#!/usr/bin/env python3
"""Module for performing pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
        kernel_shape: tuple of (kh, kw) containing the kernel shape for
                      pooling
        stride: tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode: indicates the type of pooling
            'max' indicates max pooling
            'avg' indicates average pooling

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w, c))

    # Perform pooling using only 2 loops
    for i in range(output_h):
        for j in range(output_w):
            # Calculate starting positions with stride
            start_i = i * sh
            start_j = j * sw
            # Extract the region
            region = images[:, start_i:start_i+kh, start_j:start_j+kw, :]

            # Apply pooling operation
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
