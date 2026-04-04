import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, mean_absolute_error

df = pd.read_csv('../data/processed_titanic.csv')
df = df.dropna(subset=['survived', 'age'])
df = df.fillna(0)

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

target_clf = 'survived'
X_clf = df.drop(target_clf, axis=1)
y_clf = df[target_clf]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_model = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
clf_model.fit(X_train_c, y_train_c)

y_pred_c = clf_model.predict(X_test_c)
y_proba_c = clf_model.predict_proba(X_test_c)[:, 1]

print("ЗАДАЧА КЛАССИФИКАЦИИ (ВЫЖИВАНИЕ)")
print(classification_report(y_test_c, y_pred_c))
print(f"Общая точность модели: {accuracy_score(y_test_c, y_pred_c):.2f}")

fpr, tpr, _ = roc_curve(y_test_c, y_proba_c)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('Частота ложноположительных результатов')
plt.ylabel('Истинно положительный показатель')
plt.title('ROC-кривая: Классификация выживания')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

target_reg = 'age'
X_reg = df.drop([target_reg, 'survived'], axis=1)
y_reg = df[target_reg]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = DecisionTreeRegressor(max_depth=5, random_state=42)
reg_model.fit(X_train_r, y_train_r)

y_pred_r = reg_model.predict(X_test_r)
mae = mean_absolute_error(y_test_r, y_pred_r)

print("ЗАДАЧА РЕГРЕССИИ (ВОЗРАСТ)")
print(f"Средняя абсолютная ошибка (MAE): {mae:.2f} лет")