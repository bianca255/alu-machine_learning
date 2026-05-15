# ALU Machine Learning

This repository contains implementations of machine learning algorithms and neural networks from scratch using NumPy.

## Directory Structure

```
alu-machine_learning/
├── supervised_learning/
│   └── classification/
│       ├── 0-neuron.py through 7-neuron.py
│       ├── 8-neural_network.py through 15-neural_network.py
│       ├── 16-deep_neural_network.py through 28-deep_neural_network.py
│       ├── 24-one_hot_encode.py
│       ├── 25-one_hot_decode.py
│       └── README.md
└── data/
    ├── Binary_Train.npz
    ├── Binary_Dev.npz
    └── MNIST.npz
```

## Projects

### Classification (Supervised Learning)

A comprehensive implementation of neural networks for image classification:

- **Binary Classification**: Single neuron and neural network with one hidden layer
- **Deep Networks**: Multi-layer neural networks with configurable architecture
- **Multiclass Classification**: Support for multiple output classes with softmax
- **Activation Functions**: Sigmoid and tanh for hidden layers
- **Model Persistence**: Save and load trained models using pickle

For detailed information, see [Classification README](supervised_learning/classification/README.md)

## Getting Started

1. Download the required datasets (Binary_Train.npz, Binary_Dev.npz, MNIST.npz) into the `data/` directory
2. All files are executable Python scripts with proper shebang headers
3. Each file implements one specific functionality and can be imported as a module

## Requirements

- Python 3.5+
- NumPy 1.15+
- Matplotlib (for visualization)

## Style Guide

- All code follows PEP 8 style (checked with pycodestyle 2.4)
- All modules, classes, and functions include documentation
- Code uses vectorized NumPy operations (loops minimized)
- Proper error handling with descriptive messages

## Notes

- This project is for educational purposes to understand neural networks from first principles
- All implementations use only NumPy for numerical operations
- No deep learning frameworks (TensorFlow, PyTorch, etc.) are used
