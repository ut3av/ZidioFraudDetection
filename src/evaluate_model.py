import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

DEFAULT_DATA_PATH = Path("data/processed/feature_engineered_fraud_data.csv")
DEFAULT_MODEL_PATH = Path("models/fraud_detection_model.pkl")
DEFAULT_SCALER_PATH = Path("models/scaler.pkl")
DEFAULT_REPORT_DIR = Path("reports")


def evaluate_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR
) -> dict[str, float]:
    """
    Evaluate the saved fraud detection model on the full processed dataset
    and save a structured evaluation report.
    """
    data_p = Path(data_path)
    model_p = Path(model_path)
    scaler_p = Path(scaler_path)
    report_d = Path(report_dir)

    for p, name in [(data_p, "Data file"), (model_p, "Model file"), (scaler_p, "Scaler file")]:
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found at: {p.resolve()}")

    df = pd.read_csv(data_p)
    model = joblib.load(model_p)
    scaler = joblib.load(scaler_p)

    X = df.drop(columns=["Fraud_Label"])
    y = df["Fraud_Label"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    clf_report = classification_report(y, y_pred, zero_division=0)

    print("\n--- MODEL EVALUATION SUMMARY ---")
    print(f"Total Samples Evaluated: {len(y):,}")
    print(f"Accuracy               : {acc:.4f}")
    print(f"Precision              : {prec:.4f}")
    print(f"Recall                 : {rec:.4f}")
    print(f"F1-Score               : {f1:.4f}")
    print(f"ROC-AUC Score          : {roc_auc:.4f}")
    print("\nClassification Report:\n", clf_report)
    print("Confusion Matrix:\n", cm)

    report_d.mkdir(parents=True, exist_ok=True)
    report_file = report_d / "model_evaluation.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("FRAUD DETECTION MODEL EVALUATION REPORT\n")
        f.write("=======================================\n\n")
        f.write(f"Total Records Evaluated : {len(y):,}\n")
        f.write(f"Accuracy                : {acc:.4f}\n")
        f.write(f"Precision               : {prec:.4f}\n")
        f.write(f"Recall                  : {rec:.4f}\n")
        f.write(f"F1-Score                : {f1:.4f}\n")
        f.write(f"ROC-AUC Score           : {roc_auc:.4f}\n\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("---------------------\n")
        f.write(clf_report)
        f.write("\n\nCONFUSION MATRIX\n")
        f.write("----------------\n")
        f.write(str(cm))
        f.write("\n")

    print(f"\nSaved evaluation report to: {report_file.resolve()}")
    print("--- MODEL EVALUATION COMPLETED ---\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


if __name__ == "__main__":
    evaluate_model()
