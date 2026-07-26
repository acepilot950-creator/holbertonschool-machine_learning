#!/usr/bin/env python3
"""Simple Generative Adversarial Network."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """Defines a simple Generative Adversarial Network."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """Initialize the Simple GAN."""
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

        def generator_loss(fake_output, unused):
            """Calculate the generator loss."""
            return tf.reduce_mean(
                tf.square(fake_output - tf.ones_like(fake_output))
            )

        self.generator.loss = generator_loss

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        def discriminator_loss(real_output, fake_output):
            """Calculate the discriminator loss."""
            real_loss = tf.reduce_mean(
                tf.square(real_output - tf.ones_like(real_output))
            )

            fake_loss = tf.reduce_mean(
                tf.square(fake_output + tf.ones_like(fake_output))
            )

            return real_loss + fake_loss

        self.discriminator.loss = discriminator_loss

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

        indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(indices)[:size]

        return tf.gather(
            self.real_examples,
            random_indices
        )

    def train_step(self, useless_argument):
        """Perform one complete GAN training step."""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as discr_tape:
                real_sample = self.get_real_sample()

                fake_sample = self.get_fake_sample(
                    training=False
                )

                fake_sample = tf.stop_gradient(fake_sample)

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

            discr_gradients = discr_tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables
            )

            discr_gradients_and_variables = [
                (gradient, variable)
                for gradient, variable in zip(
                    discr_gradients,
                    self.discriminator.trainable_variables
                )
                if gradient is not None
            ]

            self.discriminator.optimizer.apply_gradients(
                discr_gradients_and_variables
            )

        with tf.GradientTape() as gen_tape:
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

        gen_gradients = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )

        gen_gradients_and_variables = [
            (gradient, variable)
            for gradient, variable in zip(
                gen_gradients,
                self.generator.trainable_variables
            )
            if gradient is not None
        ]

        self.generator.optimizer.apply_gradients(
            gen_gradients_and_variables
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss
        }
