"""
main.py
Responsibility:
- Run the data pipeline
- Call models (plug-in style)
"""
import os
# Force CPU usage and suppress unnecessary logs before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.preprocessing import load_data
from src.visualize import plot_features
from src.split_scale import split_dataset, scale_dataset

# Import all models
from src.knn_model import run_knn
from src.naive_bayes_model import run_naive_bayes
from src.log_reg_model import run_log_reg
from src.svm_model import run_svm
from src.neural_net_model import run_neural_net

def main():
    # 1. Load Data
    df = load_data("data/magic04.data")

    # 2. Visualize (uncomment if needed)
    # plot_features(df)

    # 3. Split
    train_df, valid_df, test_df = split_dataset(df)

    # 4. Scale & Balance
    X_train, y_train = scale_dataset(train_df, oversample=True)
    X_valid, y_valid = scale_dataset(valid_df, oversample=False)
    X_test, y_test = scale_dataset(test_df, oversample=False)

    # Summary
    print("Dataset shapes:")
    print(f"Train: {X_train.shape} {y_train.shape}")
    print(f"Validation: {X_valid.shape} {y_valid.shape}")
    print(f"Test: {X_test.shape} {y_test.shape}\n")

    # ================================
    # RUN MODELS
    # ================================

    # 1. k-Nearest Neighbors
    run_knn(X_train, y_train, X_test, y_test, k=5)

    # 2. Naive Bayes
    run_naive_bayes(X_train, y_train, X_test, y_test)

    # 3. Logistic Regression
    run_log_reg(X_train, y_train, X_test, y_test)

    # 4. Support Vector Machine
    run_svm(X_train, y_train, X_test, y_test)

    # 5. Neural Network
    run_neural_net(X_train, y_train, X_valid, y_valid, X_test, y_test)

if __name__ == "__main__":
    main()
