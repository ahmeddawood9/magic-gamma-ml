from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def run_log_reg(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a Logistic Regression model.
    Despite the name, it is a linear model for binary classification that
    predicts the probability of a class using the Sigmoid function.
    """
    print("====================================")
    print("Logistic Regression")
    print("====================================")

    # Initialize the model. Uses the 'lbfgs' solver by default to minimize Log Loss.
    model = LogisticRegression()

    # The model learns the weights (coefficients) for each of the 10 telescope features.
    model.fit(X_train, y_train)

    # Predict class labels (0 or 1) based on a 0.5 probability threshold
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
