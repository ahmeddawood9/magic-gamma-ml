"""
main.py

Responsibility:
- Run the data pipeline
- Call models (plug-in style)

Concept:
Pipeline orchestration only
(models are plugged in as modules)
"""



from src.preprocessing import load_data
from src.visualize import plot_features
from src.split_scale import split_dataset, scale_dataset

# Models are now inside src/models/, so we import from src.models
# (Make sure these files exist, or comment them out if they don't yet)
from src.knn_model import run_knn
from src.naive_bayes_model import run_naive_bayes

def main():
    # Load dataset
    # Note: Path is relative to where you run the command (project root)
    df = load_data("data/magic04.data")

    # Visualize features
    # (Ensure visualize.py has a function named plot_features)
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

    # ================================
    # RUN MODELS
    # ================================

    # Passing the processed data to the models
    run_knn(X_train, y_train, X_test, y_test, k=5)
    run_naive_bayes(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
