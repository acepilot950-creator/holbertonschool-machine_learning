#!/usr/bin/env python3
"""
4-array.py

Selects the last 10 rows of the High and Close columns and returns them
as a NumPy array.
"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns from a DataFrame
    and converts them into a NumPy array.

    Args:
        df (pd.DataFrame): DataFrame containing High and Close columns.

    Returns:
        numpy.ndarray: Array containing the selected values.
    """
    return df[["High", "Close"]].tail(10).values
