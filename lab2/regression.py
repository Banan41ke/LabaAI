from sklearn.linear_model import LinearRegression
from utils import evaluate_regression
from sklearn.preprocessing import StandardScaler

def run_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    evaluate_regression(y_test, y_pred, model)