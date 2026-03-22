import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error
)

# ОЦЕНКА КЛАССИФИКАЦИИ
def evaluate_classification(y_test, y_pred):
    print("\nОЦЕНКА КЛАССИФИКАЦИИ")

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, y_pred))

    print("\nОтчет классификации:")
    print(classification_report(y_test, y_pred))

# ОЦЕНКА РЕГРЕССИИ
def evaluate_regression(y_test, y_pred):
    print("\nОЦЕНКА РЕГРЕССИИ")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)