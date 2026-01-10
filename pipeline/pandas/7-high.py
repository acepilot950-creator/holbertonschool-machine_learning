#!/usr/bin/env python3
"""
7-high.py

Sorts a DataFrame by the High price in descending order.
"""


def high(df):
    """
    Sorts the DataFrame by the High column in descending order.

    Args:
        df (pd.DataFrame): DataFrame containing a High column.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
