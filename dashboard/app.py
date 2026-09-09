import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💳 Financial Fraud Detection System")
st.write(
    "Machine Learning based system for detecting potentially "
    "fraudulent financial transactions."
)

st.divider()

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

MODEL_PATH = "models/fraud_detection_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/processed/feature_engineered_fraud_data.csv"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model, scaler = load_model()
df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Fraud Prediction",
        "Dataset Overview"
    ]
)

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.header("📊 Fraud Detection Dashboard")

    total_transactions = len(df)
    fraud_transactions = int(df["Fraud_Label"].sum())
    legitimate_transactions = total_transactions - fraud_transactions
    fraud_rate = (fraud_transactions / total_transactions) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraudulent Transactions",
        f"{fraud_transactions:,}"
    )

    col3.metric(
        "Legitimate Transactions",
        f"{legitimate_transactions:,}"
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    st.divider()

    st.subheader("Fraud Distribution")

    fraud_counts = df["Fraud_Label"].value_counts()

    chart_data = pd.DataFrame({
        "Transaction Type": [
            "Legitimate",
            "Fraudulent"
        ],
        "Count": [
            fraud_counts.get(0, 0),
            fraud_counts.get(1, 0)
        ]
    })

    st.bar_chart(
        chart_data.set_index("Transaction Type")
    )

# --------------------------------------------------
# FRAUD PREDICTION PAGE
# --------------------------------------------------

elif page == "Fraud Prediction":

    st.header("🔍 Transaction Fraud Prediction")

    st.write(
        "Enter transaction details below to predict whether "
        "the transaction is potentially fraudulent."
    )

    st.divider()

    # Numeric inputs
    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

    account_balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=1000.0
    )

    previous_fraud = st.number_input(
        "Previous Fraudulent Activity",
        min_value=0,
        value=0,
        step=1
    )

    daily_transactions = st.number_input(
        "Daily Transaction Count",
        min_value=0,
        value=5,
        step=1
    )

    card_age = st.number_input(
        "Card Age",
        min_value=0,
        value=365,
        step=1
    )

    # Categorical inputs
    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "Bank Transfer",
            "Online",
            "POS",
            "ATM Withdrawal"
        ]
    )

    device_type = st.selectbox(
        "Device Type",
        [
            "Mobile",
            "Tablet",
            "Laptop"
        ]
    )

    location = st.selectbox(
        "Location",
        [
            "Mumbai",
            "New York",
            "Sydney",
            "Tokyo"
        ]
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        [
            "Electronics",
            "Groceries",
            "Restaurants",
            "Travel",
            "Clothing"
        ]
    )

    card_type = st.selectbox(
        "Card Type",
        [
            "Visa",
            "Mastercard",
            "Discover"
        ]
    )

    st.divider()

    if st.button("🚨 Predict Transaction"):

        # Create basic feature dictionary
        input_data = {
            "Transaction_Amount": transaction_amount,
            "Account_Balance": account_balance,
            "Previous_Fraudulent_Activity": previous_fraud,
            "Daily_Transaction_Count": daily_transactions,
            "Card_Age": card_age
        }

        # Create dataframe
        input_df = pd.DataFrame([input_data])

        # Add engineered features
        input_df["Amount_to_Balance_Ratio"] = (
            transaction_amount /
            (account_balance + 1)
        )

        input_df["High_Amount_Flag"] = int(
            transaction_amount > 500
        )

        input_df["High_Transaction_Frequency"] = int(
            daily_transactions > 10
        )

        # Transaction type encoding
        for value in [
            "Bank Transfer",
            "Online",
            "POS"
        ]:
            input_df[
                f"Transaction_Type_{value}"
            ] = int(transaction_type == value)

        # Device encoding
        for value in [
            "Mobile",
            "Tablet"
        ]:
            input_df[
                f"Device_Type_{value}"
            ] = int(device_type == value)

        # Location encoding
        for value in [
            "Mumbai",
            "New York",
            "Sydney",
            "Tokyo"
        ]:
            input_df[
                f"Location_{value}"
            ] = int(location == value)

        # Merchant encoding
        for value in [
            "Electronics",
            "Groceries",
            "Restaurants",
            "Travel"
        ]:
            input_df[
                f"Merchant_Category_{value}"
            ] = int(merchant_category == value)

        # Card encoding
        for value in [
            "Discover",
            "Mastercard",
            "Visa"
        ]:
            input_df[
                f"Card_Type_{value}"
            ] = int(card_type == value)

        # Date-related features
        input_df["Transaction_Year"] = 2026
        input_df["Transaction_Month"] = 8
        input_df["Transaction_Day"] = 30
        input_df["Transaction_DayOfWeek"] = 6

        # Make sure feature order matches training
        expected_features = [
            col for col in df.columns
            if col != "Fraud_Label"
        ]

        input_df = input_df.reindex(
            columns=expected_features,
            fill_value=0
        )

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0]

        fraud_probability = probability[1] * 100
        legitimate_probability = probability[0] * 100

        st.divider()

        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION DETECTED"
            )

        else:

            st.success(
                "✅ LEGITIMATE TRANSACTION"
            )

        col1, col2 = st.columns(2)

        col1.metric(
            "Legitimate Probability",
            f"{legitimate_probability:.2f}%"
        )

        col2.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

# --------------------------------------------------
# DATASET OVERVIEW PAGE
# --------------------------------------------------

elif page == "Dataset Overview":

    st.header("📁 Dataset Overview")

    st.write(
        f"Dataset contains **{len(df):,} transactions** "
        f"and **{len(df.columns):,} features**."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Fraud Label Distribution")

    st.write(
        df["Fraud_Label"].value_counts()
    )