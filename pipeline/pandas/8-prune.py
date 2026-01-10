#!/usr/bin/env python3
"""
8-prune.py

Removes rows with NaN values in the Close column.
"""


def prune(df):
    """
    Removes any entries where the Close column has NaN values.

    Args:
        df (pd.DataFrame): DataFrame containing a Close column.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    return df.dropna(subset=["Close"])
