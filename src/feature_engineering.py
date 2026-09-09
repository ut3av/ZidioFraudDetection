import pandas as pd
from pathlib import Path

DEFAULT_INPUT_FILE = Path("data/processed/cleaned_fraud_data.csv")
DEFAULT_OUTPUT_FILE = Path("data/processed/feature_engineered_fraud_data.csv")


def engineer_features(
    input_path: str | Path = DEFAULT_INPUT_FILE,
    output_path: str | Path = DEFAULT_OUTPUT_FILE
) -> pd.DataFrame:
    """
    Perform feature engineering on cleaned fraud data:
    - Extracts date components (year, month, day, day of week)
    - Computes amount-to-balance ratio and threshold flags
    - One-hot encodes categorical attributes
    - Standardizes output feature matrix
    """
    input_p = Path(input_path)
    output_p = Path(output_path)

    if not input_p.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at: {input_p.resolve()}")

    df = pd.read_csv(input_p)

    print("\n--- FEATURE ENGINEERING STARTED ---")
    print(f"Input records: {df.shape[0]:,}")
    print(f"Input columns: {df.shape[1]}")

    # Temporal feature extraction
    df["Date"] = pd.to_datetime(df["Date"])
    df["Transaction_Year"] = df["Date"].dt.year
    df["Transaction_Month"] = df["Date"].dt.month
    df["Transaction_Day"] = df["Date"].dt.day
    df["Transaction_DayOfWeek"] = df["Date"].dt.dayofweek

    # Financial & behavioral ratios
    df["Amount_to_Balance_Ratio"] = df["Transaction_Amount"] / (df["Account_Balance"] + 1)
    df["High_Amount_Flag"] = (df["Transaction_Amount"] > df["Transaction_Amount"].median()).astype(int)
    df["High_Transaction_Frequency"] = (df["Daily_Transaction_Count"] > df["Daily_Transaction_Count"].median()).astype(int)

    # Remove non-predictive identifiers and original datetime string
    df = df.drop(columns=["Transaction_ID", "User_ID", "Date"], errors="ignore")

    # One-hot encoding for categorical attributes
    categorical_columns = [
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type"
    ]

    existing_cats = [col for col in categorical_columns if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True, dtype=int)

    # Reorder so target is the final column
    if "Fraud_Label" in df.columns:
        target = df["Fraud_Label"]
        features = df.drop(columns=["Fraud_Label"])
        final_df = pd.concat([features, target], axis=1)
    else:
        final_df = df

    output_p.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_p, index=False)

    print("--- FEATURE ENGINEERING COMPLETED ---")
    print(f"Output shape: {final_df.shape[0]:,} rows, {final_df.shape[1]} columns")
    print(f"Output file: {output_p.resolve()}")

    return final_df


if __name__ == "__main__":
    engineer_features()