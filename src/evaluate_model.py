import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. LOAD FEATURE-ENGINEERED DATA
# --------------------------------------------------

FILE_PATH = "data/processed/feature_engineered_fraud_data.csv"

df = pd.read_csv(FILE_PATH)

print("\n--- MODEL EVALUATION STARTED ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# --------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]


# --------------------------------------------------
# 3. LOAD TRAINED MODEL AND SCALER
# --------------------------------------------------

MODEL_PATH = "models/fraud_detection_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("\nModel loaded successfully!")
print("Scaler loaded successfully!")


# --------------------------------------------------
# 4. SCALE DATA
# --------------------------------------------------

X_scaled = scaler.transform(X)


# --------------------------------------------------
# 5. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_scaled)


# --------------------------------------------------
# 6. CALCULATE PERFORMANCE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\n--- MODEL PERFORMANCE ---")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y,
        y_pred,
        zero_division=0
    )
)

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y, y_pred))


# --------------------------------------------------
# 8. SAVE EVALUATION REPORT
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

with open("reports/model_evaluation.txt", "w") as file:

    file.write("MODEL EVALUATION REPORT\n")
    file.write("=======================\n\n")

    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1-Score : {f1:.4f}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write("=====================\n")

    file.write(
        classification_report(
            y,
            y_pred,
            zero_division=0
        )
    )

    file.write("\nCONFUSION MATRIX\n")
    file.write("================\n")

    file.write(str(confusion_matrix(y, y_pred)))


print("\nEvaluation report saved to:")
print("reports/model_evaluation.txt")

print("\n--- MODEL EVALUATION COMPLETED ---")
