from sklearn.linear_model import LogisticRegression
from utils import evaluate_classification

def run_classification(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_classification(y_test, y_pred)