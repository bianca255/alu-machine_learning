#!/usr/bin/env python3
"""Neural style transfer total-cost task."""

import numpy as np
import tensorflow as tf


class NST:
    """Add gradients to neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize style-transfer images and loss weights."""
        invalid_style = (
            not isinstance(style_image, np.ndarray)
            or style_image.ndim != 3
            or style_image.shape[2] != 3)
        if invalid_style:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        invalid_content = (
            not isinstance(content_image, np.ndarray)
            or content_image.ndim != 3
            or content_image.shape[2] != 3)
        if invalid_content:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float, np.number)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float, np.number)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        tf.compat.v1.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha, self.beta = alpha, beta
        self.model = None
        self.gram_style_features = None
        self.content_feature = None

    @staticmethod
    def scale_image(image):
        """Resize an image so its largest side is 512 and scale to [0, 1]."""
        if not isinstance(
                image,
                np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        height, width = image.shape[:2]
        scale = 512 / max(height, width)
        size = (int(height * scale), int(width * scale))
        if height >= width:
            size = (512, size[1])
        else:
            size = (size[0], 512)
        image = tf.image.resize(tf.convert_to_tensor(image, tf.float32), size,
                                method='bicubic')
        return image[tf.newaxis, ...] / 255.0

    def load_model(self):
        """Load VGG19 with average pooling and selected layer outputs."""
        original = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        def replace_pooling(layer):
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                return tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name)
            return layer

        model = tf.keras.models.clone_model(
            original, clone_function=replace_pooling)
        model.set_weights(original.get_weights())
        outputs = [model.get_layer(name).output for name in self.style_layers]
        outputs.append(model.get_layer(self.content_layer).output)
        self.model = tf.keras.Model(model.input, outputs)
        self.model.trainable = False
        return self.model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate a normalized Gram matrix from a rank-four tensor."""
        if not tf.is_tensor(input_layer) or input_layer.shape.rank != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        input_layer = input_layer[0]
        height, width = tf.shape(input_layer)[:2]
        channels = tf.shape(input_layer)[2]
        features = tf.reshape(input_layer, (height * width, channels))
        return tf.matmul(features, features, transpose_a=True) / \
            tf.cast(height * width, tf.float32)

    def generate_features(self):
        """Generate and store style Gram matrices and content features."""
        style_outputs = self.load_model()(self.style_image)
        content_outputs = self.model(self.content_image)
        self.gram_style_features = [self.gram_matrix(
            output) for output in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the cost for one style layer."""
        gram_style = self.gram_matrix(style_output)
        channels = tf.cast(tf.shape(gram_style)[0], tf.float32)
        return tf.reduce_sum(tf.square(gram_style - gram_target)
                             ) / (4.0 * tf.square(channels))

    def style_cost(self, style_outputs):
        """Calculate the total style cost for all five style layers."""
        if not isinstance(style_outputs, list) or len(style_outputs) != 5:
            raise TypeError("style_outputs must be a list of length 5")
        if self.gram_style_features is None:
            self.generate_features()
        return tf.add_n([
            self.layer_style_cost(output, target)
            for output, target in zip(style_outputs,
                                      self.gram_style_features)])

    def content_cost(self, content_output):
        """Calculate the content cost for a generated feature map."""
        if not tf.is_tensor(content_output):
            raise TypeError("content_output must be a tensor")
        if self.content_feature is None:
            self.generate_features()
        if content_output.shape != self.content_feature.shape:
            raise TypeError("content_output has the wrong shape")
        return tf.reduce_sum(tf.square(self.content_feature - content_output))

    def total_cost(self, generated_image):
        """Calculate total, content, and style costs."""
        if self.model is None or self.gram_style_features is None:
            self.generate_features()
        outputs = self.model(generated_image)
        content = self.content_cost(outputs[-1])
        style = self.style_cost(list(outputs[:-1]))
        total = self.alpha * content + self.beta * style
        return total, content, style

    def compute_grads(self, generated_image):
        """Calculate gradients and the three costs."""
        if not isinstance(generated_image, tf.Tensor):
            raise TypeError("generated_image must be a tensor")
        if generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image has the wrong shape")
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            costs = self.total_cost(generated_image)
        return tape.gradient(costs[0], generated_image), *costs
