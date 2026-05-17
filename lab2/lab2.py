import pandas as pd
from sklearn.model_selection import train_test_split

from classification import run_classification
from regression import run_regression

print("ЗАГРУЗКА ДАННЫХ")

df = pd.read_csv("../data/processed_titanic.csv")
df = df.dropna()
print(df.shape)

# КЛАССИФИКАЦИЯ
print("\nКЛАССИФИКАЦИЯ")

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

run_classification(X_train, X_test, y_train, y_test)

# РЕГРЕССИЯ
print("\nРЕГРЕССИЯ")

X = df.drop('fare', axis=1)
y = df['fare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

run_regression(X_train, X_test, y_train, y_test)