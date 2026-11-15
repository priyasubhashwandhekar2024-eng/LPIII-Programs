
# Implement K-Nearest Neighbors (KNN) on Diabetes Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load dataset
df = pd.read_csv("diabetes_data.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

# 2. Separate input features and output
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Replace invalid zero measurements with missing values
columns_to_clean = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
X[columns_to_clean] = X[columns_to_clean].replace(0, float("nan"))

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Create KNN model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# 6. Train the model
model.fit(X_train, y_train)

# 7. Predict test data
y_pred = model.predict(X_test)

# 8. Calculate evaluation metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

# 9. Display results
print("\n========== KNN RESULTS ==========")

print("\nConfusion Matrix:")
print(cm)

print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Error Rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

print("\nProgram executed successfully.")
