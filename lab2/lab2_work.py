import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('indian_roads_dataset.csv')

drop_cols = ['accident_id', 'latitude', 'longitude', 'date', 'time', 'festival', 'casualties', 'accident_severity']
df = df.drop(drop_cols, axis=1)

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(cat_cols)
#найти столбцы для категориальных данных и потом уже get_dummies и нормализация до всего должно идти если она нужна
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=['risk_score'])
y = df_encoded['risk_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
r = model.score(X_test, y_test)
y_pred = model.predict(X_test)

print("Результат линейной регрессии")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R2: {r:.4f}")