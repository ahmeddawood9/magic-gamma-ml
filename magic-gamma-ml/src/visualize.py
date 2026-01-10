"""
visualize.py

Responsibility:
- Plot feature distributions
- Compare gamma vs hadron visually

Concept:
Exploratory Data Analysis (EDA)
Helps understand feature separability before modeling
"""

import matplotlib.pyplot as plt
import os


def plot_features(df, output_dir="outputs/plots"):
    """
    Generate and save histograms for each feature.

    Parameters:
        df (DataFrame): Preprocessed dataset
        output_dir (str): Directory to save plots
    """

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    feature_columns = df.columns[:-1]  # exclude class label

    for feature in feature_columns:

        # Gamma events
        plt.hist(
            df[df["class"] == 1][feature],
            alpha=0.7,
            label="gamma",
            density=True
        )

        # Hadron events
        plt.hist(
            df[df["class"] == 0][feature],
            alpha=0.7,
            label="hadron",
            density=True
        )

        plt.title(feature)
        plt.xlabel(feature)
        plt.ylabel("Probability Density")
        plt.legend()

        # Save plot (terminal-friendly)
        plt.savefig(f"{output_dir}/{feature}.png")
        plt.close()
