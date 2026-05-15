# Binary Image Classification with Neural Networks

This project implements neural networks from scratch using NumPy for binary and multiclass image classification. The project progressively builds from a single neuron to deep neural networks with multiple hidden layers.

## Project Structure

### Task 0-7: Single Neuron (Binary Classification)
- `0-neuron.py` - Basic Neuron class with public attributes
- `1-neuron.py` - Neuron with private attributes and getters
- `2-neuron.py` - Forward propagation using sigmoid activation
- `3-neuron.py` - Cost calculation using logistic regression
- `4-neuron.py` - Evaluation (prediction and cost)
- `5-neuron.py` - Gradient descent implementation
- `6-neuron.py` - Training loop
- `7-neuron.py` - Upgraded training with verbose output and graphing

### Task 8-15: Neural Network (1 Hidden Layer, Binary Classification)
- `8-neural_network.py` - Basic NeuralNetwork with public attributes
- `9-neural_network.py` - NeuralNetwork with private attributes and getters
- `10-neural_network.py` - Forward propagation
- `11-neural_network.py` - Cost calculation
- `12-neural_network.py` - Evaluation
- `13-neural_network.py` - Gradient descent
- `14-neural_network.py` - Training loop
- `15-neural_network.py` - Upgraded training with verbose output and graphing

### Task 16-23: Deep Neural Network (Multiple Layers, Binary Classification)
- `16-deep_neural_network.py` - Basic DeepNeuralNetwork with public attributes
- `17-deep_neural_network.py` - DeepNeuralNetwork with private attributes
- `18-deep_neural_network.py` - Forward propagation
- `19-deep_neural_network.py` - Cost calculation
- `20-deep_neural_network.py` - Evaluation
- `21-deep_neural_network.py` - Gradient descent
- `22-deep_neural_network.py` - Training loop
- `23-deep_neural_network.py` - Upgraded training with verbose output and graphing

### Task 24-28: Advanced Features
- `24-one_hot_encode.py` - One-hot encoding function for multiclass labels
- `25-one_hot_decode.py` - One-hot decoding function
- `26-deep_neural_network.py` - DeepNeuralNetwork with save/load (pickle) persistence
- `27-deep_neural_network.py` - Multiclass classification support with softmax
- `28-deep_neural_network.py` - Support for different activation functions (sigmoid, tanh)

## Key Features

### Neuron Implementation
- Input validation with proper error handling
- Private attributes with property getters
- Forward propagation with sigmoid activation
- Cost calculation using binary cross-entropy
- Gradient descent optimization
- Training with configurable iterations and learning rate

### Neural Network with Hidden Layer
- Supports arbitrary number of hidden nodes
- He initialization for weights
- Forward and backward propagation
- Gradient descent for all layers

### Deep Neural Network
- Supports arbitrary number of layers
- Configurable activation functions (sigmoid, tanh)
- Softmax output for multiclass classification
- Cross-entropy loss for multiclass problems
- Model persistence (save/load with pickle)
- Verbose training output with cost tracking
- Visualization of training progress

### One-Hot Encoding/Decoding
- Converts numeric labels to one-hot matrices
- Converts one-hot matrices back to numeric labels
- Error handling for invalid inputs

## Usage

### Basic Binary Classification
```python
import numpy as np
from 7-neuron import Neuron

# Load data
X = np.random.randn(784, 100)  # 100 images of 28x28 pixels flattened
Y = np.random.randint(0, 2, (1, 100))  # Binary labels

# Create and train neuron
neuron = Neuron(784)
A, cost = neuron.train(X, Y, iterations=1000, alpha=0.01)
```

### Neural Network with Hidden Layer
```python
from 15-neural_network import NeuralNetwork

# Create and train neural network
nn = NeuralNetwork(784, 32)  # 784 input features, 32 hidden nodes
A, cost = nn.train(X, Y, iterations=1000, alpha=0.01)
```

### Deep Neural Network with Multiclass Classification
```python
from 28-deep_neural_network import DeepNeuralNetwork
from 24-one_hot_encode import one_hot_encode

# One-hot encode labels
Y_one_hot = one_hot_encode(Y, 10)  # 10 classes

# Create and train deep neural network
dnn = DeepNeuralNetwork(784, [128, 64, 10], activation='tanh')
A, cost = dnn.train(X, Y_one_hot, iterations=1000, alpha=0.01)

# Save trained model
dnn.save('model')

# Load trained model
loaded_dnn = DeepNeuralNetwork.load('model.pkl')
```

## Mathematical Concepts

### Forward Propagation
- Linear transformation: Z = W·X + b
- Activation: A = σ(Z) (sigmoid, tanh, or softmax)
- Sigmoid: σ(z) = 1/(1 + e^(-z))
- Tanh: σ(z) = (e^z - e^(-z))/(e^z + e^(-z))
- Softmax: σ(z_i) = e^(z_i) / Σ e^(z_j)

### Cost Function
- Binary: L = -(Y·log(A) + (1-Y)·log(1-A))
- Multiclass (Cross-Entropy): L = -Σ Y·log(A)

### Backpropagation
- Output layer: dZ = A - Y (for cross-entropy)
- Hidden layers: dZ = dA * σ'(Z)
- Weight update: W = W - α·dW, b = b - α·db

### Weight Initialization
- He initialization: W ~ N(0, √(2/n))

## Requirements

- Python 3.5+
- NumPy 1.15+
- Matplotlib (for graphing training progress)

## Notes

- All matrices follow shape (nodes, examples) convention
- No loops allowed except in train methods (vectorized operations preferred)
- Proper error handling with specific error messages as per requirements
- All classes and functions have documentation strings
