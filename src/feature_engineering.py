import pandas as pd
from pathlib import Path


# ---------------------------------------
# FILE PATHS
# ---------------------------------------

INPUT_FILE = "data/processed/cleaned_fraud_data.csv"
OUTPUT_FILE = "data/processed/feature_engineered_fraud_data.csv"


# ---------------------------------------
# LOAD CLEANED DATA
# ---------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\n--- FEATURE ENGINEERING STARTED ---")

print(f"Original rows: {df.shape[0]}")
print(f"Original columns: {df.shape[1]}")


# ---------------------------------------
# DATE FEATURES
# ---------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

df["Transaction_Year"] = df["Date"].dt.year
df["Transaction_Month"] = df["Date"].dt.month
df["Transaction_Day"] = df["Date"].dt.day
df["Transaction_DayOfWeek"] = df["Date"].dt.dayofweek


# ---------------------------------------
# AMOUNT-BASED FEATURES
# ---------------------------------------

df["Amount_to_Balance_Ratio"] = (
    df["Transaction_Amount"] /
    (df["Account_Balance"] + 1)
)

df["High_Amount_Flag"] = (
    df["Transaction_Amount"] >
    df["Transaction_Amount"].median()
).astype(int)


# ---------------------------------------
# TRANSACTION BEHAVIOR FEATURES
# ---------------------------------------

df["High_Transaction_Frequency"] = (
    df["Daily_Transaction_Count"] >
    df["Daily_Transaction_Count"].median()
).astype(int)


# ---------------------------------------
# DROP IDENTIFIERS / ORIGINAL DATE
# ---------------------------------------

df = df.drop(
    columns=[
        "Transaction_ID",
        "User_ID",
        "Date"
    ]
)


# ---------------------------------------
# ENCODE CATEGORICAL FEATURES
# ---------------------------------------

categorical_columns = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# ---------------------------------------
# SEPARATE TARGET
# ---------------------------------------

target = df["Fraud_Label"]

features = df.drop(
    columns=["Fraud_Label"]
)


# ---------------------------------------
# CREATE FINAL DATASET
# ---------------------------------------

final_df = pd.concat(
    [features, target],
    axis=1
)


# ---------------------------------------
# SAVE FEATURE-ENGINEERED DATA
# ---------------------------------------

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------
# RESULTS
# ---------------------------------------

print("\nNew features created:")

print("Transaction_Year")
print("Transaction_Month")
print("Transaction_Day")
print("Transaction_DayOfWeek")
print("Amount_to_Balance_Ratio")
print("High_Amount_Flag")
print("High_Transaction_Frequency")


print("\nFinal dataset:")
print(f"Rows: {final_df.shape[0]}")
print(f"Columns: {final_df.shape[1]}")


print("\nFinal column names:")
print(final_df.columns.tolist())


print("\nFraud label distribution:")
print(final_df["Fraud_Label"].value_counts())


print("\n--- FEATURE ENGINEERING COMPLETED ---")

print(
    f"Feature-engineered dataset saved to: {OUTPUT_FILE}"
)