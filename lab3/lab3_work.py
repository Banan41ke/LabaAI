import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,roc_curve,auc

df = pd.read_csv("For_EDA_dataset.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.drop(["latitude","longitude","date_added","agent","agency"], axis=1)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df,columns=cat_cols,drop_first=True)

print("Регрессия")
X = df.drop("price", axis=1)
y = df["price"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y,test_size=0.2,random_state=42)

reg_model = DecisionTreeRegressor(max_depth=5,random_state=42)
reg_model.fit(X_train_r, y_train_r)

y_pred = reg_model.predict(X_test_r)

mae = mean_absolute_error(y_test_r, y_pred)
mse = mean_squared_error(y_test_r, y_pred)
rmse = np.sqrt(mse)
r2 = reg_model.score(X_test_r, y_test_r)

print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.2f}")

print("Классификация")

df["expensive"] = (df["price"] > df["price"].median()).astype(int)

X_clf = df.drop(["price", "expensive"], axis=1)
y_clf = df["expensive"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf,y_clf,test_size=0.2,random_state=42)

clf_model = DecisionTreeClassifier(max_depth=5,criterion="gini",random_state=42)
clf_model.fit(X_train_c, y_train_c)

y_pred_c = clf_model.predict(X_test_c)
y_prob_c = clf_model.predict_proba(X_test_c)[:, 1]

acc = accuracy_score(y_test_c, y_pred_c)

print(f"Accuracy: {acc:.2f}")
fpr, tpr, _ = roc_curve(y_test_c,y_prob_c)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr,tpr,color='blue',lw=2,label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1],[0, 1],color='red',linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()