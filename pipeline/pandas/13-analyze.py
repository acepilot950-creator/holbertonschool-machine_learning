#!/usr/bin/env python3
"""
13-analyze.py

Computes descriptive statistics for all columns except Timestamp.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except the Timestamp
    column and returns them as a new DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing market data.

    Returns:
        pd.DataFrame: DataFrame with descriptive statistics.
    """
    return df.drop(columns=["Timestamp"]).describe()
