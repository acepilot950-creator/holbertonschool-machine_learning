#!/usr/bin/env python3
"""
2-from_file.py

Loads data from a file into a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pandas DataFrame.

    Args:
        filename (str): Path to the file to load.
        delimiter (str): Column separator used in the file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filename, sep=delimiter)
