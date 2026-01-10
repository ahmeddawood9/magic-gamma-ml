"""
split_scale.py

Responsibility:
- Shuffle dataset
- Split into train / validation / test
- Scale features
- Handle class imbalance (training only)

Concept:
Correct ML pipeline preparation
(This is critical for real-world ML correctness)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def split_dataset(df, train_frac=0.6, valid_frac=0.2):
    """
    Shuffle and split dataset into train, validation, and test sets.

    Returns:
        train_df, valid_df, test_df
    """

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42)

    train_end = int(train_frac * len(df))
    valid_end = int((train_frac + valid_frac) * len(df))

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]

    return train_df, valid_df, test_df


def scale_dataset(df, oversample=False):
    """
    Scale features and optionally oversample training data.

    Parameters:
        df (DataFrame)
        oversample (bool): Apply oversampling or not

    Returns:
        X (ndarray): Scaled feature matrix
        y (ndarray): Labels
    """

    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Oversampling ONLY for training data
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    return X, y
