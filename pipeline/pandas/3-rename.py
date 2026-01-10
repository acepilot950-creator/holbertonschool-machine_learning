#!/usr/bin/env python3
"""
3-rename.py

Renames the Timestamp column to Datetime and converts it to datetime format.
"""

import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts timestamp values to
    datetime values, and keeps only the Datetime and Close columns.

    Args:
        df (pd.DataFrame): DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]

    return df
