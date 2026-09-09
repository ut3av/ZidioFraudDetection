import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# -----------------------------
# FILE PATHS
# -----------------------------

INPUT_FILE = "data/processed/cleaned_fraud_data.csv"

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(INPUT_FILE)

print("\n--- EDA STARTED ---")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# -----------------------------
# BASIC INFORMATION
# -----------------------------

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# -----------------------------
# FRAUD DISTRIBUTION
# -----------------------------

print("\nFraud distribution:")
print(df["Fraud_Label"].value_counts())

print("\nFraud percentage:")
print(df["Fraud_Label"].value_counts(normalize=True) * 100)


# -----------------------------
# FRAUD DISTRIBUTION PLOT
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Fraud_Label")

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Fraud Label")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "fraud_distribution.png")

plt.close()


# -----------------------------
# TRANSACTION AMOUNT ANALYSIS
# -----------------------------

print("\nTransaction amount statistics:")
print(df["Transaction_Amount"].describe())


plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Fraud_Label", y="Transaction_Amount")

plt.title("Transaction Amount by Fraud Status")
plt.xlabel("Fraud Label")
plt.ylabel("Transaction Amount")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "transaction_amount_by_fraud.png")

plt.close()


# -----------------------------
# TRANSACTION TYPE ANALYSIS
# -----------------------------

print("\nTransaction Type distribution:")
print(df["Transaction_Type"].value_counts())


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Transaction_Type",
    hue="Fraud_Label"
)

plt.title("Transaction Type vs Fraud")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "transaction_type_vs_fraud.png")

plt.close()


# -----------------------------
# DEVICE TYPE ANALYSIS
# -----------------------------

print("\nDevice Type distribution:")
print(df["Device_Type"].value_counts())


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Device_Type",
    hue="Fraud_Label"
)

plt.title("Device Type vs Fraud")
plt.xlabel("Device Type")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "device_type_vs_fraud.png")

plt.close()


# -----------------------------
# MERCHANT CATEGORY ANALYSIS
# -----------------------------

print("\nMerchant Category distribution:")
print(df["Merchant_Category"].value_counts())


plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="Merchant_Category",
    hue="Fraud_Label"
)

plt.title("Merchant Category vs Fraud")
plt.xlabel("Merchant Category")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "merchant_category_vs_fraud.png")

plt.close()


# -----------------------------
# PREVIOUS FRAUDULENT ACTIVITY
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Previous_Fraudulent_Activity",
    hue="Fraud_Label"
)

plt.title("Previous Fraudulent Activity vs Fraud")
plt.xlabel("Previous Fraudulent Activity")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "previous_fraud_activity.png")

plt.close()


# -----------------------------
# DAILY TRANSACTION COUNT
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Fraud_Label",
    y="Daily_Transaction_Count"
)

plt.title("Daily Transaction Count by Fraud Status")
plt.xlabel("Fraud Label")
plt.ylabel("Daily Transaction Count")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "daily_transaction_count.png")

plt.close()


# -----------------------------
# ACCOUNT BALANCE
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Fraud_Label",
    y="Account_Balance"
)

plt.title("Account Balance by Fraud Status")
plt.xlabel("Fraud Label")
plt.ylabel("Account Balance")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "account_balance_by_fraud.png")

plt.close()


print("\n--- EDA COMPLETED ---")
print(f"Figures saved in: {OUTPUT_DIR}")