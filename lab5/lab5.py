import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

texts = []
sentiment = []

with open("train.ft.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        if line.startswith("__label__1"):
            sentiment.append(0)
            texts.append(line.replace("__label__1 ", "").strip())
        elif line.startswith("__label__2"):
            sentiment.append(1)
            texts.append(line.replace("__label__2 ", "").strip())

df = pd.DataFrame({"text": texts, "sentiment": sentiment})

X = df["text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(lowercase=True, max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Логистическая регрессия": LogisticRegression(
        C=1.0, max_iter=1000, random_state=42
    ),
    "Нейронная сеть (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.1,
        max_iter=500,
        learning_rate_init=0.001,
        random_state=42,
    ),
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    predictions[name] = y_pred
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }
    print(f"\n=== {name} ===")
    print(f"Точность: {results[name]['accuracy']:.2f}")
    print(results[name]["report"])

accuracies = {name: res["accuracy"] for name, res in results.items()}
plt.figure(figsize=(9, 5))
bars = plt.bar(
    accuracies.keys(), accuracies.values(), color=["#4C72B0", "#C44E52"]
)
plt.ylim(0, 1.15)
plt.title("Сравнение моделей по точности (Accuracy)", fontsize=14, pad=15)
plt.ylabel("Accuracy")
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.03,
        f"{h:.2f}",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Матрицы ошибок", fontsize=15)
for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[idx],
        cbar=False,
        xticklabels=["Негат.", "Позит."],
        yticklabels=["Негат.", "Позит."],
    )
    axes[idx].set_title(name, fontsize=11)
    axes[idx].set_xlabel("Предсказано")
    axes[idx].set_ylabel("Истина")
plt.tight_layout()
plt.show()

new_reviews = [
    "This device is absolutely spectacular! Best performance ever, highly recommend.",
    "Completely useless item. It stopped working after an hour. Extremely disappointed.",
    "The delivery was normal. The product quality is okay, not the worst but nothing special.",
]
new_tfidf = vectorizer.transform(new_reviews)
mlp = models["Нейронная сеть (MLP)"]
preds = mlp.predict(new_tfidf)
probs = mlp.predict_proba(new_tfidf)
for text, pred, prob in zip(new_reviews, preds, probs):
    label = "Положительный" if pred == 1 else "Отрицательный"
    confidence = max(prob)
    print(f"Отзыв: {text[:40]}...")
    print(f"Тональность: {label} (уверенность: {confidence:.0%})")
    print()
