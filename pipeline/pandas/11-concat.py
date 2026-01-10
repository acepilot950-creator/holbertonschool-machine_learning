#!/usr/bin/env python3
"""
11-concat.py

Concatenates two DataFrames (bitstamp and coinbase) after indexing by Timestamp,
keeping bitstamp rows up to a specified cutoff timestamp and adding source keys.
"""


def concat(df1, df2):
    """
    Indexes both DataFrames on Timestamp, selects rows from df2 up to and
    including timestamp 1417411920, concatenates them above df1, and adds
    keys ('bitstamp', 'coinbase').

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame with a Timestamp column.
        df2 (pd.DataFrame): Bitstamp DataFrame with a Timestamp column.

    Returns:
        pd.DataFrame: Concatenated DataFrame with a MultiIndex (source, Timestamp).
    """
    index = __import__('10-index').index
    pd = __import__('pandas')

    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[df2.index <= 1417411920]

    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
