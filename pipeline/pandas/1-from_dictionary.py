#!/usr/bin/env python3
"""
1-from_dictionary.py

Creates a pandas DataFrame from a dictionary with labeled rows and columns.
"""

import pandas as pd

# create the data dictionary
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# create the DataFrame with custom row labels
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
