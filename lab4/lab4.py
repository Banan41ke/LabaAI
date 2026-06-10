import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")

# Заменяем физически невозможные нули на NaN, чтобы правильно посчитать медианы
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)

# Заполняем пропуски медианным значением в зависимости от целевого класса (0 или 1)
for col in cols_with_zeros:
    df[col] = df[col].fillna(df.groupby("Outcome")[col].transform("median"))

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=150, max_depth=10, min_samples_split=4, oob_score=True, random_state=42
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("=== Случайный лес ===")
print(f"OOB Accuracy: {rf.oob_score_:.3f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(classification_report(y_test, y_pred_rf))

ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.2, random_state=42)
ada.fit(X_train_scaled, y_train)
y_pred_ada = ada.predict(X_test_scaled)

print("=== AdaBoost ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada):.3f}")
print(classification_report(y_test, y_pred_ada))

gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42
)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)

print("=== Градиентный бустинг ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.3f}")
print(classification_report(y_test, y_pred_gb))

plt.figure(figsize=(8, 6))
for name, scores, color in [
    ("Random Forest", rf.predict_proba(X_test_scaled)[:, 1], "blue"),
    ("AdaBoost", ada.predict_proba(X_test_scaled)[:, 1], "green"),
    ("Gradient Boosting", gb.predict_proba(X_test_scaled)[:, 1], "red"),
]:
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_score = roc_auc_score(y_test, scores)
    plt.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc_score:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Случайный")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC-кривые моделей ансамблей")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
