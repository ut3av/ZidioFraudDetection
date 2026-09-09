import pandas as pd

FILE_PATH = "data/raw/synthetic_fraud_dataset1.csv"


def load_data():
    df = pd.read_csv(FILE_PATH)
    return df


if __name__ == "__main__":
    df = load_data()

    print("\nDataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFraud distribution:")
    print(df["Fraud_Label"].value_counts())


    print("\nFraud distribution:")
    print("\nAll column names:")
print(df.columns.tolist())
