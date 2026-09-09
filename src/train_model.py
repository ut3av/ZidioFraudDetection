import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD FEATURE-ENGINEERED DATA
# --------------------------------------------------

FILE_PATH = "data/processed/feature_engineered_fraud_data.csv"

df = pd.read_csv(FILE_PATH)

print("\n--- MODEL TRAINING STARTED ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# --------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])


# --------------------------------------------------
# 4. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 5. TRAIN LOGISTIC REGRESSION MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

print("\nModel training completed!")


# --------------------------------------------------
# 6. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------------------------
# 7. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_detection_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved to:")
print("models/fraud_detection_model.pkl")

print("Scaler saved to:")
print("models/scaler.pkl")

print("\n--- MODEL TRAINING COMPLETED ---")