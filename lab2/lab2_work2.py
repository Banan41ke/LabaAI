import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("For_EDA_dataset.csv")

print(df.head())
print(df.isnull().sum())

df = df.drop(["province_name","latitude","longitude","date_added","agent", "agency"],axis=1)

print(df.select_dtypes(include="object").columns)
df = pd.get_dummies(df,columns=['property_type', 'location', 'city', 'purpose'],drop_first=True)

scaler = MinMaxScaler()
df[["Area_in_Marla"]] = scaler.fit_transform(df[["Area_in_Marla"]])

X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()


model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

