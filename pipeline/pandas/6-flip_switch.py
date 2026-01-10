#!/usr/bin/env python3
"""
6-flip_switch.py

Sorts a DataFrame in reverse chronological order and transposes it.
"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order and transposes it.

    Args:
        df (pd.DataFrame): DataFrame to transform.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    df = df.sort_index(ascending=False)
    return df.T
