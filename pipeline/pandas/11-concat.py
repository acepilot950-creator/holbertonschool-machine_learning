#!/usr/bin/env python3
"""
11-concat.py

Concatenates bitstamp data (up to a cutoff timestamp) above coinbase data,
after indexing both DataFrames by the Timestamp column and adding source keys.
"""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes both dataframes on Timestamp, selects rows from df2 up to and
    including timestamp 1417411920, concatenates them above df1, and adds
    keys to label sources.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame containing a Timestamp column.
        df2 (pd.DataFrame): Bitstamp DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: Concatenated DataFrame with a MultiIndex.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[df2.index <= 1417411920]

    return pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )
