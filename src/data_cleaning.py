import pandas as pd
from pathlib import Path

DEFAULT_INPUT_PATH = Path("data/raw/synthetic_fraud_dataset1.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/cleaned_fraud_data.csv")


def clean_data(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> pd.DataFrame:
    """
    Clean raw transaction dataset by removing duplicates, converting date types,
    and dropping null records.
    """
    input_p = Path(input_path)
    output_p = Path(output_path)

    if not input_p.is_file():
        raise FileNotFoundError(f"Input file not found at: {input_p.resolve()}")

    df = pd.read_csv(input_p)

    print("\n--- DATA CLEANING STARTED ---")
    print(f"Initial row count    : {df.shape[0]:,}")
    print(f"Initial column count : {df.shape[1]}")

    duplicates = df.duplicated().sum()
    print(f"Duplicates identified: {duplicates:,}")
    df = df.drop_duplicates()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    missing_count = df.isnull().sum().sum()
    print(f"Missing values found : {missing_count:,}")
    df = df.dropna().reset_index(drop=True)

    output_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_p, index=False)

    print("--- DATA CLEANING COMPLETED ---")
    print(f"Cleaned row count    : {df.shape[0]:,}")
    print(f"Cleaned column count : {df.shape[1]}")
    print(f"Saved cleaned data to: {output_p.resolve()}")

    return df


if __name__ == "__main__":
    clean_data()