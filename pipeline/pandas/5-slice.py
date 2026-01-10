#!/usr/bin/env python3
"""
5-slice.py

Extracts specific columns from a DataFrame and selects every 60th row.
"""


def slice(df):
    """
    Extracts the High, Low, Close, and Volume_(BTC) columns and selects
    every 60th row.

    Args:
        df (pd.DataFrame): DataFrame containing the required columns.

    Returns:
        pd.DataFrame: The sliced DataFrame.
    """
    df = df[["High", "Low", "Close", "Volume_(BTC)"]]
    return df.iloc[::60]
