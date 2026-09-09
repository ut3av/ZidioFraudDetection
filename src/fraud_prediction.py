import joblib
import pandas as pd
from pathlib import Path

DEFAULT_DATA_PATH = Path("data/processed/feature_engineered_fraud_data.csv")
DEFAULT_MODEL_PATH = Path("models/fraud_detection_model.pkl")
DEFAULT_SCALER_PATH = Path("models/scaler.pkl")
DEFAULT_OUTPUT_PATH = Path("outputs/prediction_result.csv")


def predict_sample(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sample_index: int = 0
) -> pd.DataFrame:
    """
    Run single or sample transaction inference using the trained model pipeline.
    """
    data_p = Path(data_path)
    model_p = Path(model_path)
    scaler_p = Path(scaler_path)
    output_p = Path(output_path)

    for p, name in [(data_p, "Data file"), (model_p, "Model file"), (scaler_p, "Scaler file")]:
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found at: {p.resolve()}")

    model = joblib.load(model_p)
    scaler = joblib.load(scaler_p)
    df = pd.read_csv(data_p)

    sample_transaction = df.drop(columns=["Fraud_Label"]).iloc[[sample_index]]
    actual_label = df["Fraud_Label"].iloc[sample_index] if "Fraud_Label" in df.columns else None

    sample_scaled = scaler.transform(sample_transaction)
    prediction = int(model.predict(sample_scaled)[0])
    probabilities = model.predict_proba(sample_scaled)[0]

    print("\n--- TRANSACTION INFERENCE RESULT ---")
    if actual_label is not None:
        print(f"Ground Truth Label  : {'FRAUD (1)' if actual_label == 1 else 'LEGITIMATE (0)'}")
    print(f"Model Prediction    : {'FRAUD (1)' if prediction == 1 else 'LEGITIMATE (0)'}")
    print(f"Legitimate Confidence: {probabilities[0]:.2%}")
    print(f"Fraud Confidence     : {probabilities[1]:.2%}")

    result_df = sample_transaction.copy()
    result_df["Actual_Fraud_Label"] = actual_label
    result_df["Predicted_Fraud_Label"] = prediction
    result_df["Fraud_Probability"] = probabilities[1]
    result_df["Legitimate_Probability"] = probabilities[0]

    output_p.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_p, index=False)
    print(f"Prediction output saved to: {output_p.resolve()}")
    print("------------------------------------\n")

    return result_df


if __name__ == "__main__":
    predict_sample()