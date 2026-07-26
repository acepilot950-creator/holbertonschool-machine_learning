#!/usr/bin/env python3
"""Defines a simple Generative Adversarial Network."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """Represents a simple Generative Adversarial Network."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """Initialize the Simple GAN model."""
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        self.generator.loss = (
            lambda x, y:
            tf.keras.losses.MeanSquaredError()(
                x,
                tf.ones_like(x)
            )
        )

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        self.discriminator.loss = (
            lambda x, y:
            tf.keras.losses.MeanSquaredError()(
                x,
                tf.ones_like(x)
            )
            + tf.keras.losses.MeanSquaredError()(
                y,
                -tf.ones_like(y)
            )
        )

        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate and return a batch of fake samples."""
        if size is None:
            size = self.batch_size

        latent_sample = self.latent_generator(size)

        return self.generator(
            latent_sample,
            training=training
        )

    def get_real_sample(self, size=None):
        """Return a random batch of real samples."""
        if size is None:
            size = self.batch_size

        sorted_indices = tf.range(
            tf.shape(self.real_examples)[0]
        )

        random_indices = tf.random.shuffle(
            sorted_indices
        )[:size]

        return tf.gather(
            self.real_examples,
            random_indices
        )

    def train_step(self, useless_argument):
        """Perform one complete GAN training step."""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample()

                real_output = self.discriminator(
                    real_sample,
                    training=True
                )

                fake_output = self.discriminator(
                    fake_sample,
                    training=True
                )

                discr_loss = self.discriminator.loss(
                    real_output,
                    fake_output
                )

            discr_gradients = tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables
            )

            self.discriminator.optimizer.apply_gradients(
                zip(
                    discr_gradients,
                    self.discriminator.trainable_variables
                )
            )

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(
                training=True
            )

            fake_output = self.discriminator(
                fake_sample,
                training=False
            )

            gen_loss = self.generator.loss(
                fake_output,
                None
            )

        gen_gradients = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(
                gen_gradients,
                self.generator.trainable_variables
            )
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss
        }
