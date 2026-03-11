#!/usr/bin/env python3
"""Module for performing convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding
    
    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution
        padding: tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    
    # Pad the images
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )
    
    # Calculate output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    
    # Initialize output array
    output = np.zeros((m, output_h, output_w))
    
    # Perform convolution using only 2 loops
    for i in range(output_h):
        for j in range(output_w):
            # Extract the region and perform element-wise multiplication
            output[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )
    
    return output
