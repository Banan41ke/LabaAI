from sklearn.linear_model import LinearRegression
from utils import evaluate_regression

def run_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_regression(y_test, y_pred)