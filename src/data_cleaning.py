import pandas as pd

INPUT_PATH = "data/raw/synthetic_fraud_dataset1.csv"
OUTPUT_PATH = "data/processed/cleaned_fraud_data.csv"


def clean_data():

    # Load raw dataset
    df = pd.read_csv(INPUT_PATH)

    print("\n--- DATA CLEANING STARTED ---")
    print(f"Original rows: {df.shape[0]}")
    print(f"Original columns: {df.shape[1]}")

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")

    df = df.drop_duplicates()

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Check missing values
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # Remove rows with missing values
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- DATA CLEANING COMPLETED ---")
    print(f"Cleaned rows: {df.shape[0]}")
    print(f"Cleaned columns: {df.shape[1]}")
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    clean_data()