"""
preprocessing.py

Responsibility:
- Load raw dataset
- Assign column names
- Convert class labels to numeric

Concept:
Raw data  →  clean, ML-ready dataframe
"""

import pandas as pd


def load_data(path):
    """
    Load MAGIC Gamma Telescope dataset and preprocess labels.

    Parameters:
        path (str): Path to magic04.data

    Returns:
        df (DataFrame): Preprocessed dataframe
    """

    # Dataset has no header, so we define column names manually
    cols = [
        "fLength", "fWidth", "fSize", "fConc", "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
        "class"
    ]

    # Read CSV file
    df = pd.read_csv(path, names=cols)

    # Convert class labels:
    # 'g' (gamma)   → 1
    # 'h' (hadron)  → 0
    df["class"] = (df["class"] == "g").astype(int)

    return df
