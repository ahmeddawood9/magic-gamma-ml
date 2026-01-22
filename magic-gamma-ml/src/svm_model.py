from sklearn.svm import SVC
from sklearn.metrics import classification_report

def run_svm(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a Support Vector Machine (SVM) classifier.
    SVM finds the optimal hyperplane that maximizes the margin between classes.
    """
    print("====================================")
    print("Support Vector Machine (SVM)")
    print("====================================")

    # Initialize the Support Vector Classifier.
    # By default, it uses the RBF (Radial Basis Function) kernel, which allows
    # for non-linear decision boundaries by projecting data into higher dimensions.
    svm_model = SVC()

    # Fit the model: The algorithm identifies 'support vectors' (critical boundary points)
    # to define the maximum-margin hyperplane.
    svm_model = svm_model.fit(X_train, y_train)

    # Generate predictions on the unseen test set
    y_pred = svm_model.predict(X_test)

    # Output detailed metrics: Precision, Recall, and F1-Score
    print(classification_report(y_test, y_pred))
