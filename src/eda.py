import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DEFAULT_INPUT_FILE = Path("data/processed/cleaned_fraud_data.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/figures")


def run_eda(
    input_path: str | Path = DEFAULT_INPUT_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> None:
    """
    Generate exploratory data analysis visualizations and summary statistics.
    """
    input_p = Path(input_path)
    output_d = Path(output_dir)

    if not input_p.is_file():
        raise FileNotFoundError(f"Dataset not found at: {input_p.resolve()}")

    output_d.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_p)

    print("\n--- EXPLORATORY DATA ANALYSIS STARTED ---")
    print(f"Dataset Dimension: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Set visualization theme
    sns.set_theme(style="whitegrid")

    # 1. Target Distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="Fraud_Label")
    plt.title("Distribution of Fraud vs. Non-Fraud Transactions", fontsize=12, fontweight="bold")
    plt.xlabel("Fraud Label (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(output_d / "fraud_distribution.png", dpi=300)
    plt.close()

    # 2. Transaction Amount by Class
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Fraud_Label", y="Transaction_Amount")
    plt.title("Transaction Amount Distribution by Fraud Status", fontsize=12, fontweight="bold")
    plt.xlabel("Fraud Label (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Amount ($)")
    plt.tight_layout()
    plt.savefig(output_d / "transaction_amount_by_fraud.png", dpi=300)
    plt.close()

    # 3. Transaction Type vs Fraud
    if "Transaction_Type" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Transaction_Type", hue="Fraud_Label")
        plt.title("Fraud Incidence by Transaction Type", fontsize=12, fontweight="bold")
        plt.xlabel("Transaction Type")
        plt.ylabel("Transaction Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_d / "transaction_type_vs_fraud.png", dpi=300)
        plt.close()

    # 4. Device Type vs Fraud
    if "Device_Type" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Device_Type", hue="Fraud_Label")
        plt.title("Fraud Incidence by Device Type", fontsize=12, fontweight="bold")
        plt.xlabel("Device Type")
        plt.ylabel("Transaction Count")
        plt.tight_layout()
        plt.savefig(output_d / "device_type_vs_fraud.png", dpi=300)
        plt.close()

    # 5. Merchant Category vs Fraud
    if "Merchant_Category" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x="Merchant_Category", hue="Fraud_Label")
        plt.title("Fraud Incidence by Merchant Category", fontsize=12, fontweight="bold")
        plt.xlabel("Merchant Category")
        plt.ylabel("Transaction Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_d / "merchant_category_vs_fraud.png", dpi=300)
        plt.close()

    # 6. Previous Fraud History vs Current Status
    if "Previous_Fraudulent_Activity" in df.columns:
        plt.figure(figsize=(7, 5))
        sns.countplot(data=df, x="Previous_Fraudulent_Activity", hue="Fraud_Label")
        plt.title("Current Fraud Status by Historical Fraud Flag", fontsize=12, fontweight="bold")
        plt.xlabel("Previous Fraud Activity (0 = No, 1 = Yes)")
        plt.ylabel("Transaction Count")
        plt.tight_layout()
        plt.savefig(output_d / "previous_fraud_activity.png", dpi=300)
        plt.close()

    # 7. Daily Transaction Frequency
    if "Daily_Transaction_Count" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Fraud_Label", y="Daily_Transaction_Count")
        plt.title("Daily Transaction Count by Fraud Status", fontsize=12, fontweight="bold")
        plt.xlabel("Fraud Label (0 = Legitimate, 1 = Fraud)")
        plt.ylabel("Daily Transaction Count")
        plt.tight_layout()
        plt.savefig(output_d / "daily_transaction_count.png", dpi=300)
        plt.close()

    # 8. Account Balance
    if "Account_Balance" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Fraud_Label", y="Account_Balance")
        plt.title("Account Balance by Fraud Status", fontsize=12, fontweight="bold")
        plt.xlabel("Fraud Label (0 = Legitimate, 1 = Fraud)")
        plt.ylabel("Account Balance ($)")
        plt.tight_layout()
        plt.savefig(output_d / "account_balance_by_fraud.png", dpi=300)
        plt.close()

    print(f"Generated 8 analytical plots saved to: {output_d.resolve()}")
    print("--- EXPLORATORY DATA ANALYSIS COMPLETED ---")


if __name__ == "__main__":
    run_eda()