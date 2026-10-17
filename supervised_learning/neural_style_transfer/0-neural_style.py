#!/usr/bin/env python3
"""Neural style transfer implementation."""

import numpy as np
import tensorflow as tf


class NST:
    """Represent the data and operations used by neural style transfer."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1',
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize style-transfer images and loss weights."""
        tf.compat.v1.enable_eager_execution()
        image_message = (
            "style_image must be a numpy.ndarray with shape (h, w, 3)"
        )
        if not isinstance(style_image, np.ndarray) or (
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(image_message)
        image_message = (
            "content_image must be a numpy.ndarray with shape (h, w, 3)"
        )
        if not isinstance(content_image, np.ndarray) or (
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(image_message)
        if not isinstance(alpha, (int, float, np.number)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float, np.number)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Resize an image so its largest side is 512 and scale to [0, 1]."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or (
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
        height, width = image.shape[:2]
        scale = 512 / max(height, width)
        size = (int(height * scale), int(width * scale))
        if height >= width:
            size = (512, size[1])
        else:
            size = (size[0], 512)
        scaled = tf.image.resize(
            tf.convert_to_tensor(image, dtype=tf.float32), size,
            method='bicubic'
        )
        return scaled[tf.newaxis, ...] / 255.0
