import pandas as pd
import joblib


# --------------------------------------------------
# 1. LOAD TRAINED MODEL AND SCALER
# --------------------------------------------------

MODEL_PATH = "models/fraud_detection_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("\n--- FRAUD PREDICTION SYSTEM ---")
print("Model loaded successfully!")
print("Scaler loaded successfully!")


# --------------------------------------------------
# 2. LOAD FEATURE-ENGINEERED DATA
# --------------------------------------------------

FILE_PATH = "data/processed/feature_engineered_fraud_data.csv"

df = pd.read_csv(FILE_PATH)


# --------------------------------------------------
# 3. SELECT ONE TRANSACTION FOR DEMONSTRATION
# --------------------------------------------------

sample_transaction = df.drop(columns=["Fraud_Label"]).iloc[[0]]

actual_label = df["Fraud_Label"].iloc[0]


# --------------------------------------------------
# 4. SCALE TRANSACTION
# --------------------------------------------------

sample_scaled = scaler.transform(sample_transaction)


# --------------------------------------------------
# 5. MAKE PREDICTION
# --------------------------------------------------

prediction = model.predict(sample_scaled)[0]

probability = model.predict_proba(sample_scaled)[0]


# --------------------------------------------------
# 6. DISPLAY PREDICTION
# --------------------------------------------------

print("\n--- TRANSACTION PREDICTION ---")

print(f"Actual Fraud Label: {actual_label}")

if prediction == 1:
    print("Prediction: FRAUDULENT TRANSACTION")
else:
    print("Prediction: LEGITIMATE TRANSACTION")

print(f"Legitimate Probability: {probability[0]:.2%}")
print(f"Fraud Probability: {probability[1]:.2%}")


# --------------------------------------------------
# 7. SAVE PREDICTION
# --------------------------------------------------

result = sample_transaction.copy()

result["Predicted_Fraud_Label"] = prediction
result["Fraud_Probability"] = probability[1]

result.to_csv(
    "outputs/prediction_result.csv",
    index=False
)

print("\nPrediction saved to:")
print("outputs/prediction_result.csv")

print("\n--- FRAUD PREDICTION COMPLETED ---")