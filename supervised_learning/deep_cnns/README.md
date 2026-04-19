# Deep Convolutional Architectures

This project focuses on the implementation of advanced convolutional neural network (CNN) building blocks using TensorFlow/Keras. The goal is to understand how modern architectures are constructed and how different convolutional strategies improve feature extraction.

## Inception Block

The Inception block is based on the architecture introduced in *"Going Deeper with Convolutions"* (2014). It allows the network to process input data at multiple scales simultaneously.

The block consists of four parallel paths:
- 1x1 convolution
- 1x1 convolution followed by 3x3 convolution
- 1x1 convolution followed by 5x5 convolution
- 3x3 max pooling followed by 1x1 convolution

The outputs of all paths are concatenated along the channel axis.

## Key Concepts

- **1x1 Convolutions** are used for dimensionality reduction and computational efficiency.
- **Multiple filter sizes (3x3, 5x5)** allow capturing features at different spatial scales.
- **Max pooling branch** provides robustness and feature generalization.
- **Concatenation** combines all extracted features into a single output tensor.

## Requirements

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Code follows `pycodestyle (2.11.1)`
- All modules and functions include proper documentation
