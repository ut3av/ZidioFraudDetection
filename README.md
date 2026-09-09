# Financial Fraud Detection System

An end-to-end Machine Learning pipeline and analytical dashboard designed to detect fraudulent financial transactions with high precision and recall using ensemble classification methods.

---

## Project Overview

Financial fraud represents a critical operational risk for payment processors, banks, and e-commerce platforms. This project delivers a production-grade machine learning solution that automates:
- Raw transaction ingestion and automated data cleansing
- Temporal and behavioral feature engineering (ratios, threshold indicators, categorical encoding)
- Exploratory data analysis (EDA) with statistical anomaly visualizations
- Supervised model training using a balanced Random Forest classifier
- Comprehensive classification evaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Interactive web-based decision support system powered by Streamlit

---

## Architecture and Workflow

```
+-------------------------------------------------------------+
|                      Raw Transactions                       |
|           (data/raw/synthetic_fraud_dataset1.csv)           |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       1. Data Cleaning                      |
|                     (src/data_cleaning.py)                  |
|          - Duplicate removal, type casting, NA purging      |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   2. Feature Engineering                    |
|                (src/feature_engineering.py)                 |
|     - Date decomposition (Year, Month, Day, DayOfWeek)      |
|     - Amount-to-balance ratio, High-frequency flags         |
|     - One-hot categorical encoding                          |
+-------------------------------------------------------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+-----------------------+             +-----------------------+
|  3. Visual Analytics  |             |   4. Model Training   |
|     (src/eda.py)      |             |  (src/train_model.py) |
| - High-res charts in  |             | - Random Forest (300) |
|   outputs/figures/    |             | - Scaler & Model PKL  |
+-----------------------+             +-----------------------+
                                                  |
                                                  v
                                      +-----------------------+
                                      | 5. Model Evaluation   |
                                      | (src/evaluate_model)  |
                                      | - Reports & Metrics   |
                                      +-----------------------+
                                                  |
                                                  v
                                      +-----------------------+
                                      | 6. Interactive App    |
                                      |  (dashboard/app.py)   |
                                      | - Streamlit Dashboard |
                                      +-----------------------+
```

---

## Project Structure

```
ZidioFraudDetection/
|-- dashboard/
|   `-- app.py                  # Interactive Streamlit dashboard and inference UI
|-- data/
|   |-- raw/
|   |   `-- synthetic_fraud_dataset1.csv   # Raw transaction records (50,000 samples)
|   `-- processed/
|       |-- cleaned_fraud_data.csv         # Cleaned transaction dataset
|       `-- feature_engineered_fraud_data.csv # Final feature matrix for modeling
|-- models/
|   |-- fraud_detection_model.pkl          # Trained Random Forest classifier
|   `-- scaler.pkl                         # Fitted StandardScaler instance
|-- outputs/
|   |-- figures/                           # Generated EDA visualization charts
|   |   |-- account_balance_by_fraud.png
|   |   |-- daily_transaction_count.png
|   |   |-- device_type_vs_fraud.png
|   |   |-- fraud_distribution.png
|   |   |-- merchant_category_vs_fraud.png
|   |   |-- previous_fraud_activity.png
|   |   |-- transaction_amount_by_fraud.png
|   |   `-- transaction_type_vs_fraud.png
|   `-- prediction_result.csv              # Sample prediction output
|-- reports/
|   `-- model_evaluation.txt               # Quantitative performance report
|-- src/
|   |-- data_cleaning.py                   # Data ingestion and cleaning routines
|   |-- data_loading.py                    # Dataset inspection utilities
|   |-- eda.py                             # Automated EDA plot generation
|   |-- evaluate_model.py                  # Evaluation metrics computation
|   |-- feature_engineering.py             # Feature extraction and encoding
|   |-- fraud_prediction.py                # Standalone sample inference script
|   `-- train_model.py                     # Model training and artifact export
|-- .gitignore                             # Git exclusion configuration
|-- main.py                                # End-to-end pipeline CLI orchestrator
|-- README.md                              # Project documentation
`-- requirements.txt                       # Python dependencies
```

---

## Model Performance

The Random Forest Classifier (configured with 300 estimators and balanced class weighting) achieves the following metrics on the evaluation dataset:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 92.90% |
| **Precision (Fraud Class)** | 95.04% |
| **Recall (Fraud Class)** | 82.18% |
| **F1-Score** | 0.8814 |
| **Macro Average F1** | 0.9200 |

### Confusion Matrix
```
                  Predicted Legitimate    Predicted Fraud
Actual Legitimate        33,244                 689
Actual Fraud              2,863              13,204
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/ut3av/ZidioFraudDetection.git
cd ZidioFraudDetection
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Run the Full ML Pipeline
To execute data cleaning, feature engineering, exploratory analysis, model training, and evaluation in one step:
```bash
python main.py --step all
```

### 2. Execute Individual Pipeline Steps
```bash
# Clean raw transaction records
python main.py --step clean

# Generate engineered feature matrix
python main.py --step features

# Generate exploratory analysis charts
python main.py --step eda

# Train the Random Forest classifier
python main.py --step train

# Evaluate the model and save report
python main.py --step evaluate

# Test single transaction inference
python main.py --step predict
```

### 3. Launch the Interactive Dashboard
Start the Streamlit web interface for interactive risk assessment, dataset exploration, and visual analytics:
```bash
streamlit run dashboard/app.py
```

---

## Key Features & Input Attributes

The model predicts transaction risk using the following primary and engineered attributes:
- **Transaction Amount**: Value of the transaction in USD
- **Account Balance**: Remaining balance in the originating account
- **Amount-to-Balance Ratio**: Relative transaction load against available liquidity
- **Daily Transaction Frequency**: Total transaction volume within the past 24 hours
- **Previous Fraudulent Activity**: Historical anomaly indicator
- **Card Age**: Longevity of the payment card in days
- **Transaction Type**: ATM Withdrawal, Bank Transfer, Online, POS
- **Device & Location**: Access device (Laptop, Mobile, Tablet) and transaction city
- **Merchant Category**: Electronics, Groceries, Restaurants, Travel, Clothing

---

## License

This project is developed for educational and professional demonstration purposes under Zidio Development.
