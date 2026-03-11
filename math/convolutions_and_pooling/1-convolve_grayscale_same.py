#!/usr/bin/env python3
"""Module for performing same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for 'same' convolution
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the images
    padded_images = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    # Initialize output array (same size as input)
    output = np.zeros((m, h, w))

    # Perform convolution using only 2 loops
    for i in range(h):
        for j in range(w):
            # Extract the region and perform element-wise multiplication
            output[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return output
