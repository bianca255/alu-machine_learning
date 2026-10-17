#!/usr/bin/env python3
"""Neural style transfer Gram-matrix task."""

import numpy as np
import tensorflow as tf


class NST:
    """Add Gram matrices and feature generation to neural style transfer."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1',
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize style-transfer images and loss weights."""
        tf.compat.v1.enable_eager_execution()
        if not isinstance(style_image, np.ndarray) or (
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or (
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float, np.number)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float, np.number)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.generate_features()

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
            tf.convert_to_tensor(image, dtype=tf.float32),
            size, method='bicubic'
        )
        return scaled[tf.newaxis, ...] / 255.0

    def load_model(self):
        """Load VGG19 with average pooling and selected layer outputs."""
        original = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )

        def replace_pooling(layer):
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                return tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size, strides=layer.strides,
                    padding=layer.padding, name=layer.name
                )
            return layer

        model = tf.keras.models.clone_model(
            original, clone_function=replace_pooling
        )
        model.set_weights(original.get_weights())
        outputs = [model.get_layer(name).output for name in self.style_layers]
        outputs.append(model.get_layer(self.content_layer).output)
        model = tf.keras.Model(model.input, outputs)
        model.trainable = False
        self.model = model
        return model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate a normalized Gram matrix from a rank-four tensor."""
        if not tf.is_tensor(input_layer) or input_layer.shape.rank != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        input_layer = input_layer[0]
        height = tf.shape(input_layer)[0]
        width = tf.shape(input_layer)[1]
        channels = tf.shape(input_layer)[2]
        features = tf.reshape(input_layer, (height * width, channels))
        gram = tf.matmul(features, features, transpose_a=True)
        return gram / tf.cast(height * width, tf.float32)

    def generate_features(self):
        """Generate and store style Gram matrices and content features."""
        self.model = self.load_model()
        style_outputs = self.model(self.style_image)
        content_outputs = self.model(self.content_image)
        self.gram_style_features = [
            self.gram_matrix(style_output)
            for style_output in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]
