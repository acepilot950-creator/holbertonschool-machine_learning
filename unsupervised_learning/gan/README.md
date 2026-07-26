# Generative Adversarial Networks

## Description

This project introduces the fundamental concepts of Generative Adversarial
Networks (GANs).

A GAN consists of two neural networks:

- **Generator**: creates fake samples from randomly generated latent vectors.
- **Discriminator**: attempts to distinguish real samples from generated
  samples.

The generator and discriminator are trained adversarially. The discriminator
learns to identify fake samples, while the generator learns to produce samples
that can fool the discriminator.

The first task implements a simple GAN by subclassing
`tensorflow.keras.Model` and overriding its `train_step` method.

## Learning Objectives

At the end of this project, I should be able to explain:

- What a Generative Adversarial Network is
- What a generator is
- What a discriminator is
- How the generator and discriminator are trained
- What adversarial training means
- What a latent vector is
- What a latent space is
- How TensorFlow's `GradientTape` is used to calculate gradients
- How to implement a custom Keras training step
- How GAN losses and optimizers are defined

## GAN Architecture

A simple GAN contains two main components.

### Generator

The generator receives a latent vector \(z\) and transforms it into a fake
sample:

```text
latent vector z -> Generator -> fake sample
