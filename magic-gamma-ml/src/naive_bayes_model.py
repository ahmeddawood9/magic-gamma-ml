from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


def run_naive_bayes(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Gaussian Naive Bayes classifier
    """

    # Create Naive Bayes model
    nb_model = GaussianNB()

    # Fit model (learn mean, variance, priors)
    nb_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = nb_model.predict(X_test)

    # Print evaluation report
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred))
