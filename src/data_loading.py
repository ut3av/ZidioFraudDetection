import pandas as pd
from pathlib import Path

DEFAULT_INPUT_PATH = Path("data/raw/synthetic_fraud_dataset1.csv")


def load_raw_data(file_path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """
    Load raw transaction data from a CSV file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found at: {path.resolve()}")
    return pd.read_csv(path)


def main() -> None:
    df = load_raw_data()
    print("\n--- DATA LOADING SUMMARY ---")
    print(f"Total Records : {df.shape[0]:,}")
    print(f"Total Columns : {df.shape[1]}")
    print("\nColumn List:")
    for col in df.columns:
        print(f" - {col} ({df[col].dtype})")
    print("\nMissing Values per Column:")
    print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else " None")
    print("\nFraud Class Distribution:")
    print(df["Fraud_Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
