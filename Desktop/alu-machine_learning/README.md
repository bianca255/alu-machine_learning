# ALU Machine Learning Classification

This repository contains a comprehensive implementation of neural network classification algorithms from scratch using only NumPy.

## Classification Tasks (0-28)

The classification module implements binary and multiclass classification using neural networks of increasing complexity:

### Tasks 0-7: Single Neuron (Binary Classification)
- Basic neuron with forward propagation
- Cost calculation using binary cross-entropy
- Gradient descent optimization
- Full training loop with visualization

### Tasks 8-15: Neural Network (1 Hidden Layer)
- Single hidden layer neural network
- Forward propagation through two layers
- Backpropagation for parameter updates
- Training with optional visualization

### Tasks 16-23: Deep Neural Network (Multiple Layers)
- Multi-layer neural network with configurable architecture
- He weight initialization
- Deep forward propagation with caching
- Deep backpropagation algorithm
- Training with cost visualization

### Tasks 24-28: Advanced Features
- One-hot encoding/decoding for multiclass
- Model persistence (pickle serialization)
- Multiclass classification with softmax
- Configurable activation functions (sigmoid/tanh)

## File Requirements

All files follow strict specifications:
- Python 3.5+ executable scripts
- Only NumPy import (plus matplotlib for visualization)
- PEP 8 compliant (pycodestyle 2.4)
- Comprehensive docstrings
- Proper error handling

## Mathematical Foundations

### Activation Functions
- **Sigmoid**: σ(z) = 1 / (1 + e^-z)
- **Tanh**: tanh(z) = (e^z - e^-z) / (e^z + e^-z)  
- **Softmax**: e^z / Σ(e^z) for multiclass output

### Cost Functions
- **Binary Cross-Entropy**: -(Y·log(A) + (1-Y)·log(1-A)) / m
- **Categorical Cross-Entropy**: -Σ(Y·log(A)) / m

### Optimization
- Gradient Descent with configurable learning rate (alpha)
- Backpropagation for parameter computation
- Weight updates: W = W - α·dW, b = b - α·db

## Code Structure

```
supervised_learning/classification/
├── 0-7-neuron.py          (8 task files for Neuron)
├── 8-15-neural_network.py (8 task files for NeuralNetwork)
├── 16-23-deep_neural_network.py (8 task files for DeepNeuralNetwork)
├── 24-one_hot_encode.py
├── 25-one_hot_decode.py
├── 26-deep_neural_network.py (with save/load)
├── 27-deep_neural_network.py (multiclass support)
├── 28-deep_neural_network.py (activation functions)
└── README.md
```

## Implementation Notes

- Vectorized NumPy operations minimize loops
- Matrix operations follow (features/nodes, examples) convention
- Weight initialization uses He method for deeper networks
- Cost computation includes numerical stability safeguards
- Model persistence uses Python pickle serialization
