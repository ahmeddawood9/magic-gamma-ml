from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def run_knn(X_train, y_train, X_test, y_test, k=5):
    """
    Train and evaluate K-Nearest Neighbors classifier
    """

    # Create KNN model
    knn_model = KNeighborsClassifier(n_neighbors=k)

    # Fit model (store training data)
    knn_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn_model.predict(X_test)

    # Print evaluation report
    print(f"\nKNN (k={k}) Classification Report:")
    print(classification_report(y_test, y_pred))
