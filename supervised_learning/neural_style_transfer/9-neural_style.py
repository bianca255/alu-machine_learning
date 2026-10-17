#!/usr/bin/env python3
"""Neural style transfer image-generation task."""

import numpy as np
import tensorflow as tf


class NST:
    """Generate an image with Adam and retain the best result."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1',
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize style-transfer images and loss weights."""
        tf.compat.v1.enable_eager_execution()
        self._validate_image(style_image, 'style_image')
        self._validate_image(content_image, 'content_image')
        if not isinstance(alpha, (int, float, np.number)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float, np.number)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha, self.beta = alpha, beta
        self.model = None
        self.gram_style_features = None
        self.content_feature = None
        self.generate_features()

    @staticmethod
    def _validate_image(image, name):
        """Validate an unscaled RGB image."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                "{} must be a numpy.ndarray with shape (h, w, 3)".format(
                    name))

    @staticmethod
    def scale_image(image):
        """Resize an image so its largest side is 512 and scale to [0, 1]."""
        NST._validate_image(image, 'image')
        height, width = image.shape[:2]
        scale = 512 / max(height, width)
        size = (int(height * scale), int(width * scale))
        if height >= width:
            size = (512, size[1])
        else:
            size = (size[0], 512)
        image = tf.image.resize(
            tf.convert_to_tensor(image, dtype=tf.float32),
            size, method='bicubic')
        return image[tf.newaxis, ...] / 255.0

    def load_model(self):
        """Load VGG19 with average pooling and selected layer outputs."""
        original = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        def replace_pooling(layer):
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                return tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size, strides=layer.strides,
                    padding=layer.padding, name=layer.name)
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
        shape = tf.shape(input_layer)
        features = tf.reshape(input_layer, (shape[0] * shape[1], shape[2]))
        gram = tf.matmul(features, features, transpose_a=True)
        return gram[tf.newaxis, ...] / tf.cast(
            shape[0] * shape[1], tf.float32)

    def generate_features(self):
        """Generate and store style Gram matrices and content features."""
        self.load_model()
        style_outputs = self.model(self.style_image)
        content_outputs = self.model(self.content_image)
        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the cost for one style layer."""
        if not tf.is_tensor(style_output):
            raise TypeError("style_output must be a tensor")
        if not tf.is_tensor(gram_target):
            raise TypeError("gram_target must be a tensor")
        gram_style = self.gram_matrix(style_output)
        channels = tf.cast(tf.shape(gram_style)[1], tf.float32)
        return tf.reduce_sum(tf.square(gram_style - gram_target)) / (
            4.0 * tf.square(channels))

    def style_cost(self, style_outputs):
        """Calculate the total style cost for all five style layers."""
        if not isinstance(style_outputs, list) or len(style_outputs) != 5:
            raise TypeError(
                "style_outputs must be a list with a length of 5")
        if self.gram_style_features is None:
            self.generate_features()
        weights = [1 / len(style_outputs)] * len(style_outputs)
        return tf.add_n([
            weight * self.layer_style_cost(output, target)
            for weight, output, target in zip(
                weights, style_outputs, self.gram_style_features)])

    def content_cost(self, content_output):
        """Calculate the content cost for a generated feature map."""
        if not isinstance(content_output, tf.Tensor):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape))
        if self.content_feature is None:
            self.generate_features()
        if content_output.shape != self.content_feature.shape:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape))
        return tf.reduce_sum(tf.square(self.content_feature - content_output))

    def total_cost(self, generated_image):
        """Calculate total, content, and style costs."""
        valid_tensor = tf.is_tensor(generated_image)
        if not valid_tensor:
            valid_tensor = isinstance(generated_image, tf.Variable)
        if not valid_tensor:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape))
        if generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape))
        if self.model is None or self.gram_style_features is None:
            self.generate_features()
        outputs = self.model(generated_image)
        content = self.content_cost(outputs[-1])
        style = self.style_cost(list(outputs[:-1]))
        total = self.alpha * content + self.beta * style
        return total, content, style

    def compute_grads(self, generated_image):
        """Calculate gradients and the three costs."""
        valid_tensor = tf.is_tensor(generated_image)
        if not valid_tensor:
            valid_tensor = isinstance(generated_image, tf.Variable)
        if not valid_tensor:
            raise TypeError("generated_image must be a tensor")
        if generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image has the wrong shape")
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            costs = self.total_cost(generated_image)
        return tape.gradient(costs[0], generated_image), *costs

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generate the lowest-cost image using Adam."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and (step <= 0 or step >= iterations):
            raise ValueError(
                "step must be positive and less than iterations")
        if not isinstance(lr, (int, float, np.number)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not 0 <= beta1 <= 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not 0 <= beta2 <= 1:
            raise ValueError("beta2 must be in the range [0, 1]")
        generated = tf.Variable(self.content_image)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2)
        best_image, best_cost = None, float('inf')
        for iteration in range(iterations):
            gradients, total, content, style = self.compute_grads(generated)
            if (iteration == 0 or
                    (step is not None and iteration % step == 0) or
                    iteration == iterations - 1):
                print(
                    "Cost at iteration {}: {}, content {}, style {}".format(
                        iteration, total, content, style))
            cost = float(total.numpy())
            if cost < best_cost:
                best_cost, best_image = cost, generated.numpy().copy()
            optimizer.apply_gradients([(gradients, generated)])
            generated.assign(tf.clip_by_value(generated, 0.0, 1.0))
        return best_image, best_cost
