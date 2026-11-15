
# Bank Customer Churn Prediction using Neural Network

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Read the dataset
df = pd.read_csv("Churn_Modelling.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 2. Separate features and target
# Remove unnecessary columns
X = df.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
y = df["Exited"]

# Identify numerical and categorical columns
numeric_features = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary"
]

categorical_features = ["Geography", "Gender"]

# Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 3. Normalize and encode the data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# 4. Build the improved Neural Network model
model = Pipeline([
    ("preprocessing", preprocessor),

    ("classifier", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        alpha=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Predict the test data
y_pred = model.predict(X_test)

# 5. Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:", accuracy * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
