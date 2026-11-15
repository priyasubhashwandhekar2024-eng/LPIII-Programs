
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("emails.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

# 2. Remove missing values
df = df.dropna()

# The last column contains the target labels:
# 0 = Not Spam, 1 = Spam
target_column = df.columns[-1]

# Remove email ID column and target column from features
X = df.iloc[:, 1:-1]
y = df[target_column]

# Convert labels to numeric
y = pd.to_numeric(y, errors="coerce")
X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows containing invalid values
valid_rows = y.notna() & X.notna().all(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

print("\nClass distribution:")
print(y.value_counts().sort_index())

# 3. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Initialize classifiers
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel="linear")
}

results = []

# 6. Train and evaluate both models
for name, model in models.items():
    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    ))

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

# 7. Compare model performance
results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

best_model = results_df.loc[
    results_df["F1 Score"].idxmax(), "Model"
]
print("\nBest model based on F1 Score:", best_model)
