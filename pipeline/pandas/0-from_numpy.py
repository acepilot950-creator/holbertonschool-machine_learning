i#!/usr/bin/env python3
"""
0-from_numpy.py

Creates a pandas DataFrame from a NumPy ndarray with columns labeled A-Z.
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    The columns are labeled in alphabetical order and capitalized (A-Z).

    Args:
        array (np.ndarray): The array to convert into a DataFrame.

    Returns:
        pd.DataFrame: The newly created DataFrame.
    """
    n_cols = array.shape[1]
    columns = [chr(ord('A') + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=columns)
