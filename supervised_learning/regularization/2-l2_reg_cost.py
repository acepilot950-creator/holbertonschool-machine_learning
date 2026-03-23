#!/usr/bin/env python3
"""L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """calculates the cost of a neural network with L2 regularization"""
    return tf.stack([cost + loss for loss in model.losses])
