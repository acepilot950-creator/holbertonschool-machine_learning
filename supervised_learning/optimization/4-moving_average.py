#!/usr/bin/env python3
"""Module that calculates the weighted moving average of a dataset."""


def moving_average(data, beta):
    """Calculate the weighted moving average with bias correction.

    Args:
        data (list): List of numerical values.
        beta (float): Weight for the moving average.

    Returns:
        list: Moving averages of the data.
    """
    v = 0
    moving_avg = []

    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        v_corrected = v / (1 - beta ** (i + 1))
        moving_avg.append(v_corrected)

    return moving_avg
