import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# -----------------------------
# 1. Load Dataset
# -----------------------------
data_path = "../dataset/transport_data.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print(df.head())


# -----------------------------
# 2. Separate Features & Target
# -----------------------------
X = df.drop("Actual_Delay_Min", axis=1)
y = df["Actual_Delay_Min"]


# -----------------------------
# 3. Identify Columns
# -----------------------------
categorical_features = [
    "Route",
    "Traffic",
    "Weather",
    "Peak_Hour"
]

numeric_features = [
    "Distance_KM",
    "Previous_Delay_Min"
]


# -----------------------------
# 4. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# -----------------------------
# 5. Create ML Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# -----------------------------
# 6. Create Pipeline
# -----------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------
# 7. Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 8. Train Model
# -----------------------------
print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model trained successfully!")


# -----------------------------
# 9. Evaluate Model
# -----------------------------
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n----- MODEL PERFORMANCE -----")
print(f"Mean Absolute Error: {mae:.2f} minutes")
print(f"R2 Score: {r2:.2f}")


# -----------------------------
# 10. Save Model
# -----------------------------
model_path = "transport_delay_model.pkl"

joblib.dump(pipeline, model_path)

print("\nModel saved successfully!")
print(f"Saved at: {os.path.abspath(model_path)}")