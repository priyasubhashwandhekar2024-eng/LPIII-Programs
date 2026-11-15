
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load Kaggle dataset
df = pd.read_csv("uber.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)

# Select useful columns
df = df[["fare_amount", "pickup_longitude", "pickup_latitude",
         "dropoff_longitude", "dropoff_latitude", "passenger_count"]]

# Convert columns to numeric and remove missing values
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

# Remove invalid values and extreme fares
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

# Calculate distance in kilometers
lat1 = np.radians(df["pickup_latitude"])
lat2 = np.radians(df["dropoff_latitude"])
lon1 = np.radians(df["pickup_longitude"])
lon2 = np.radians(df["dropoff_longitude"])

a = (np.sin((lat2-lat1)/2)**2 +
     np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)

df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
df = df[df["distance_km"] > 0].copy()

print("\nRows after cleaning:", len(df))

if len(df) < 20:
    raise ValueError("Too few valid rows. Check your dataset.")

# Limit dataset size to make training faster
if len(df) > 50000:
    df = df.sample(50000, random_state=42)

# Features and target
X = df[["pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "passenger_count", "distance_km"]]
y = df["fare_amount"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

print("\nMODEL EVALUATION")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"\n{name}")
    print("R2 Score:", round(r2_score(y_test, predictions), 4))
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mean_absolute_error(y_test, predictions), 2))

# Example prediction using the Random Forest model
model = models["Random Forest"]
sample = X_test.iloc[[0]]
predicted_fare = model.predict(sample)[0]

print("\nEXAMPLE FARE PREDICTION")
print("Distance (km):", round(sample["distance_km"].iloc[0], 2))
print("Predicted fare ($):", round(predicted_fare, 2))
