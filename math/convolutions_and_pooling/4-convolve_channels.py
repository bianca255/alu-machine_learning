#!/usr/bin/env python3
"""Module for performing convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels
    
    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple images
        kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel
                for the convolution
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
    
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride
    
    # Determine padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    
    # Pad the images (only height and width dimensions, not channels)
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    # Calculate output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1
    
    # Initialize output array
    output = np.zeros((m, output_h, output_w))
    
    # Perform convolution using only 2 loops
    for i in range(output_h):
        for j in range(output_w):
            # Calculate starting positions with stride
            start_i = i * sh
            start_j = j * sw
            # Extract the region and perform element-wise multiplication
            # Sum across height, width, and channels
            output[:, i, j] = np.sum(
                padded_images[:, start_i:start_i+kh, start_j:start_j+kw, :] * kernel,
                axis=(1, 2, 3)
            )
    
    return output
