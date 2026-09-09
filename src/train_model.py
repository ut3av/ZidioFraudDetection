import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

DEFAULT_DATA_PATH = Path("data/processed/feature_engineered_fraud_data.csv")
DEFAULT_MODEL_DIR = Path("models")


def train_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    random_state: int = 42
) -> tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a Random Forest classifier for fraud detection,
    scale numerical features, evaluate performance on a hold-out test set,
    and persist the trained model and scaler artifacts.
    """
    data_p = Path(data_path)
    model_d = Path(model_dir)

    if not data_p.is_file():
        raise FileNotFoundError(f"Feature engineered data not found at: {data_p.resolve()}")

    df = pd.read_csv(data_p)

    print("\n--- MODEL TRAINING STARTED ---")
    print(f"Dataset Shape: {df.shape[0]:,} records, {df.shape[1]} features")

    # Feature and Target Split
    X = df.drop(columns=["Fraud_Label"])
    y = df["Fraud_Label"]

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"Training Set : {X_train.shape[0]:,} samples")
    print(f"Test Set     : {X_test.shape[0]:,} samples")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Initialization and Training
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    print("Training Random Forest Classifier (n_estimators=300, class_weight=balanced)...")
    model.fit(X_train_scaled, y_train)

    # Evaluation on Test Partition
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- HOLD-OUT TEST EVALUATION ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Persist Artifacts
    model_d.mkdir(parents=True, exist_ok=True)
    model_file = model_d / "fraud_detection_model.pkl"
    scaler_file = model_d / "scaler.pkl"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    print("\n--- ARTIFACTS SAVED ---")
    print(f"Model  : {model_file.resolve()}")
    print(f"Scaler : {scaler_file.resolve()}")
    print("--- MODEL TRAINING COMPLETED ---\n")

    return model, scaler


if __name__ == "__main__":
    train_model()