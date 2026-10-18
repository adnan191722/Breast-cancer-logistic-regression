"""
Predictive Modeling Project: Breast Cancer Diagnosis using Logistic Regression
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
)

OUT_DIR = "images"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load data -------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# In this dataset: 0 = malignant, 1 = benign
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
print("Shape:", df.shape)
print("Class distribution:\n", df["diagnosis"].value_counts())

# 2. Data cleaning / preprocessing ----------------------------------------
print("\nMissing values:", int(df.isnull().sum().sum()))
before = len(df)
df = df.drop_duplicates()
print("Duplicates removed:", before - len(df))

X = df[data.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. Train model -----------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

# 4. Evaluate --------------------------------------------------------------
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_prob),
}
print("\n=== Test Set Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("\n", classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

with open("results.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

# 5. Visualizations --------------------------------------------------------
sns.set_theme(style="whitegrid")

# 5a. Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# 5b. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {metrics['ROC AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
plt.close()

# 5c. Top feature coefficients
coef = pd.Series(model.coef_[0], index=data.feature_names)
top = coef.reindex(coef.abs().sort_values(ascending=False).head(10).index).sort_values()
plt.figure(figsize=(7, 5))
colors = ["#d9534f" if v < 0 else "#5cb85c" for v in top.values]
top.plot(kind="barh", color=colors)
plt.xlabel("Coefficient (standardized features)")
plt.title("Top 10 Features Influencing Prediction\n(positive = more likely Benign)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
plt.close()

print(f"\nGraphs saved in '{OUT_DIR}/'")
