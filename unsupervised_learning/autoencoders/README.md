# Autoencoders

## Description

This project introduces autoencoders, a type of artificial neural network used to learn efficient representations of data.

An autoencoder consists of two main components:

- **Encoder** — compresses the input data into a lower-dimensional latent representation.
- **Decoder** — reconstructs the original input from the latent representation.

The model is trained by comparing the reconstructed output with the original input and minimizing the reconstruction loss.

In this project, autoencoders are implemented using TensorFlow and Keras.

## Learning Objectives

By the end of this project, I should be able to explain:

- What an autoencoder is
- The purpose of an encoder and a decoder
- What a latent space representation is
- What a bottleneck is
- How a vanilla autoencoder works
- How autoencoders are trained
- The difference between vanilla and variational autoencoders
- Which loss functions are used for autoencoders
- How to implement autoencoders using TensorFlow and Keras

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- TensorFlow 2.15
- Keras
- Pycodestyle 2.11.1

All files are interpreted or compiled on Ubuntu 20.04 LTS using Python 3.9.

All Python files must:

- Start with the line:

```python
#!/usr/bin/env python3
