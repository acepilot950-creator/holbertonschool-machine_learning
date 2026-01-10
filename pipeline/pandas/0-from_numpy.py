#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    # number of columns
    n_cols = array.shape[1]

    # create column labels: A, B, C, ...
    columns = [chr(ord('A') + i) for i in range(n_cols)]

    # create the DataFrame
    df = pd.DataFrame(array, columns=columns)

    return df
