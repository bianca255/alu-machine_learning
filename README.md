# ALU Machine Learning - Classification Module

This repository contains implementations of neural network classification algorithms from scratch using NumPy.

## Overview

A comprehensive implementation of binary and multiclass neural networks, from a single neuron to deep multi-layer networks, with support for training visualization and model persistence.

## Tasks 0-28

### Binary Classification (Tasks 0-15)
- **Tasks 0-7**: Single Neuron with forward propagation, cost computation, gradient descent, and training with visualization
- **Tasks 8-15**: Neural Network with one hidden layer for binary classification

### Deep Learning (Tasks 16-23)
- Multi-layer deep neural networks with configurable architecture
- He weight initialization
- Forward propagation with activation caching
- Backpropagation through multiple layers
- Training with cost visualization

### Multiclass & Advanced Features (Tasks 24-28)
- One-hot encoding and decoding
- Model serialization (save/load)
- Softmax activation for multiclass output
- Cross-entropy loss for multiclass problems
- Configurable activation functions (sigmoid/tanh)

## Requirements

- Python 3.5+
- NumPy 1.15+
- Matplotlib (optional, for visualization)

## Implementation Details

- All code uses only NumPy for numerical computations
- Vectorized operations minimize loops for performance
- Proper error handling with descriptive messages
- PEP 8 compliant code
- Comprehensive docstrings

## Mathematical Concepts

- Sigmoid and Tanh activation functions
- Softmax for multiclass classification
- Binary cross-entropy loss
- Categorical cross-entropy loss
- Gradient descent optimization
- Backpropagation algorithm
- He initialization for deep networks
