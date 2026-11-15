
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. LOAD DATASET
df = pd.read_csv("uber.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

# 2. DATA PREPROCESSING
df = df[["fare_amount", "pickup_longitude", "pickup_latitude",
         "dropoff_longitude", "dropoff_latitude", "passenger_count"]]

# Convert columns into numeric values
df = df.apply(pd.to_numeric, errors="coerce")

# Remove missing values
df = df.dropna()

# 3. IDENTIFY AND REMOVE OUTLIERS
df = df[
    (df["fare_amount"] > 0) &
    (df["fare_amount"] <= 200) &
    (df["passenger_count"] >= 1) &
    (df["passenger_count"] <= 6) &
    (df["pickup_longitude"].between(-75, -72)) &
    (df["dropoff_longitude"].between(-75, -72)) &
    (df["pickup_latitude"].between(40, 42)) &
    (df["dropoff_latitude"].between(40, 42))
].copy()

print("\nRows after removing outliers:", len(df))

# 4. CALCULATE DISTANCE USING HAVERSINE FORMULA
lat1 = np.radians(df["pickup_latitude"])
lat2 = np.radians(df["dropoff_latitude"])
lon1 = np.radians(df["pickup_longitude"])
lon2 = np.radians(df["dropoff_longitude"])

a = (np.sin((lat2 - lat1) / 2)**2 +
     np.cos(lat1) * np.cos(lat2) *
     np.sin((lon2 - lon1) / 2)**2)

df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

# Remove zero-distance rides
df = df[df["distance_km"] > 0].copy()

print("\nRows after cleaning:", len(df))

# Check minimum records
if len(df) < 20:
    raise ValueError("Too few valid rows. Check your dataset.")

# Reduce dataset size for faster training
if len(df) > 50000:
    df = df.sample(50000, random_state=42)

# 5. CORRELATION ANALYSIS
print("\nCORRELATION MATRIX:")
print(df.corr().round(2))

# 6. SELECT FEATURES AND TARGET
X = df[["pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "passenger_count", "distance_km"]]

y = df["fare_amount"]

# 7. SPLIT DATASET INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))

# 8. IMPLEMENT MACHINE LEARNING MODELS
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# 9. TRAIN AND EVALUATE MODELS
print("\nMODEL EVALUATION")

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predict fare
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    print("\n", name)
    print("R2 Score:", round(r2, 4))
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))

# 10. EXAMPLE FARE PREDICTION
model = models["Random Forest"]

sample = X_test.iloc[[0]]

predicted_fare = model.predict(sample)[0]

print("\nEXAMPLE FARE PREDICTION")
print("Distance (km):", round(sample["distance_km"].iloc[0], 2))
print("Predicted fare ($):", round(predicted_fare, 2))
