"""
main.py

Responsibility:
- Run the data pipeline
- No models yet

Concept:
Pipeline orchestration only
(models will be plugged in later)
"""

from preprocessing import load_data
from visualize import plot_features
from split_scale import split_dataset, scale_dataset


def main():
    # Load and preprocess dataset
    df = load_data("data/magic04.data")

    # Visualize feature distributions
    plot_features(df)

    # Split dataset
    train_df, valid_df, test_df = split_dataset(df)

    # Scale datasets
    X_train, y_train = scale_dataset(train_df, oversample=True)
    X_valid, y_valid = scale_dataset(valid_df, oversample=False)
    X_test, y_test = scale_dataset(test_df, oversample=False)

    # Dataset summary
    print("Dataset shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_valid.shape, y_valid.shape)
    print("Test:", X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()
