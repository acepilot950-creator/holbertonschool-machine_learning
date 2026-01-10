#!/usr/bin/env python3
"""
10-index.py

Sets the Timestamp column as the index of a DataFrame.
"""


def index(df):
    """
    Sets the Timestamp column as the index of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    return df.set_index("Timestamp")
