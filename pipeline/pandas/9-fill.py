#!/usr/bin/env python3
"""
9-fill.py

Cleans and fills missing values in a DataFrame according to specific rules.
"""


def fill(df):
    """
    Removes the Weighted_Price column, fills missing values in Close with the
    previous value, fills missing values in Open, High, and Low with the
    corresponding Close value, and sets missing volume values to 0.

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # remove the Weighted_Price column
    df = df.drop(columns=["Weighted_Price"])

    # fill missing Close values with the previous row's value
    df["Close"] = df["Close"].fillna(method="ffill")

    # fill Open, High, and Low missing values with Close from the same row
    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])

    # set missing volume values to 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
