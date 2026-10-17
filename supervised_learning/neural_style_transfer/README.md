# Neural Style Transfer

This directory implements the ALU neural style transfer assignment with
TensorFlow eager execution. Each numbered module defines an independently
importable `NST` class and accumulates the methods introduced by the previous
task.

The implementation scales images to `[0, 1]`, extracts features from a frozen
VGG19 model using average pooling, computes normalized Gram matrices, and
optimizes a generated image with Adam. Tasks 0 through 9 return total,
content, and style costs. Task 10 adds total-variation regularization and
returns the variation cost as a fourth value.

## Files

- `0-neural_style.py`: initialization and public VGG layer definitions
- `1-neural_style.py`: proportional bicubic image scaling
- `2-neural_style.py`: VGG19 model loading
- `3-neural_style.py`: Gram matrices and feature generation
- `4-neural_style.py`: one-layer style cost
- `5-neural_style.py`: aggregate style cost
- `6-neural_style.py`: content cost
- `7-neural_style.py`: total cost
- `8-neural_style.py`: gradients
- `9-neural_style.py`: Adam image generation
- `10-neural_style.py`: variation cost and variational total cost