i#!/usr/bin/env python3
"""
12-hierarchy.py

Creates a MultiIndex DataFrame where Timestamp is the first level,
combining bitstamp and coinbase data for a specific time range
in chronological order.
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Indexes both DataFrames on Timestamp, selects rows from timestamps
    1417411980 to 1417417980 inclusive, concatenates them with keys
    (bitstamp, coinbase), rearranges the MultiIndex so Timestamp is
    the first level, and sorts the result chronologically.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame with a Timestamp column.
        df2 (pd.DataFrame): Bitstamp DataFrame with a Timestamp column.

    Returns:
        pd.DataFrame: Concatenated DataFrame with MultiIndex
        (Timestamp, source).
    """
    start = 1417411980
    end = 1417417980

    df1 = index(df1).loc[start:end]
    df2 = index(df2).loc[start:end]

    df = pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )

    df = df.swaplevel(0, 1)

    sources = pd.CategoricalIndex(
        df.index.get_level_values(1),
        categories=["bitstamp", "coinbase"],
        ordered=True
    )

    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values(0), sources],
        names=["Timestamp", None]
    )

    return df.sort_index()
