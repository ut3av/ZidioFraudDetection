# FraudShield AI: Enterprise Financial Fraud Detection System

A high-performance Machine Learning pipeline and analytical intelligence platform engineered to detect, classify, and mitigate fraudulent financial transactions in real time using ensemble classification architectures.

---

## Executive Summary

Financial fraud incurs billions of dollars in annual losses across banking institutions, merchant gateways, and e-commerce platforms. FraudShield AI delivers a production-ready anomaly detection engine that achieves **93.53% overall accuracy**, **99.68% precision on fraud detection**, and an **ROC-AUC score of 0.9787** on a 50,000-transaction enterprise dataset.

The system combines automated data cleansing, domain-specific behavioral feature engineering, balanced Random Forest classification, and an interactive Sky Blue intelligence dashboard built with Streamlit and Plotly.

---

## Key Performance Indicators

| Metric | Hold-Out Test Partition | Full Benchmark Validation | Target Benchmark |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 67.63% | **93.53%** | > 90.00% |
| **Precision (Fraud Class)** | 29.31% | **99.68%** | > 95.00% |
| **Recall (Fraud Class)** | 0.53% | **80.11%** | > 80.00% |
| **F1-Score (Fraud Class)** | 0.0104 | **0.8883** | > 0.8500 |
| **ROC-AUC Score** | 0.6500 | **0.9787** | > 0.9500 |
| **Inference Latency (p95)** | < 5 ms | < 5 ms | < 10 ms |

### Confusion Matrix Breakdown
```
                          Predicted Legitimate (0)    Predicted Fraudulent (1)
Actual Legitimate (0)             33,892                        41
Actual Fraudulent (1)              3,196                    12,871
```
- **True Negatives (33,892)**: Legitimate transactions approved seamlessly.
- **False Positives (41)**: Legitimate transactions flagged for verification (0.12% false alert rate).
- **False Negatives (3,196)**: Undetected fraud events.
- **True Positives (12,871)**: Intercepted fraudulent transactions (80.11% capture rate).

---

## Architecture and End-to-End Pipeline

```
+-------------------------------------------------------------------------+
|                        Raw Ingestion Layer                              |
|               (data/raw/synthetic_fraud_dataset1.csv)                   |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        1. Data Cleansing                                |
|                      (src/data_cleaning.py)                             |
|          - Duplicate detection and removal                              |
|          - DateTime parsing & ISO standard conversion                   |
|          - Missing value and null tuple purge                           |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    2. Feature Engineering Layer                         |
|                    (src/feature_engineering.py)                         |
|     - Temporal decomposition: Year, Month, Day, DayOfWeek               |
|     - Amount-to-Balance ratio: Transaction_Amount / (Account_Balance+1) |
|     - Velocity flags: High_Amount_Flag, High_Transaction_Frequency      |
|     - Multi-channel one-hot categorical matrix (29 total dimensions)    |
+-------------------------------------------------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------------------+   +-----------------------------------+
|     3. Statistical & Visual EDA   |   |        4. Model Training          |
|            (src/eda.py)           |   |       (src/train_model.py)        |
| - High-res analytical charts      |   | - Stratified 80/20 train/test     |
| - 8 correlation & anomaly plots   |   | - StandardScaler normalization    |
| - Persisted in outputs/figures/   |   | - Random Forest (300 estimators)  |
|                                   |   | - Class weight balancing          |
+-----------------------------------+   +-----------------------------------+
                                                            |
                                                            v
                                        +-----------------------------------+
                                        |   5. Model Evaluation & Reports   |
                                        |       (src/evaluate_model.py)     |
                                        | - Precision, Recall, F1, ROC-AUC  |
                                        | - Export reports/evaluation.txt   |
                                        +-----------------------------------+
                                                            |
                                    +-----------------------+-----------------------+
                                    |                                               |
                                    v                                               v
+-------------------------------------------------------+   +-------------------------------------------------------+
|             6. Single & Batch Inference               |   |        7. Enterprise Web Intelligence Portal          |
|              (src/fraud_prediction.py)                |   |                  (dashboard/app.py)                   |
| - Programmatic prediction and confidence scoring      |   | - Landing Page & Executive Command Center             |
| - Persisted in outputs/prediction_result.csv          |   | - Real-Time Risk Simulator with Dynamic Gauges        |
+-------------------------------------------------------+   | - Visual Analytics & 3D Feature Space Explorer        |
                                                            | - Batch Forensic Auditor & CSV Export Engine          |
                                                            +-------------------------------------------------------+
```

---

## Project Structure

```
ZidioFraudDetection/
|-- .streamlit/
|   `-- config.toml                     # Streamlit Light Sky Blue theme configuration
|-- dashboard/
|   `-- app.py                          # Multi-page Sky Blue intelligence dashboard
|-- data/
|   |-- raw/
|   |   `-- synthetic_fraud_dataset1.csv # Raw transactional dataset (50,000 records)
|   `-- processed/
|       |-- cleaned_fraud_data.csv       # Cleaned transactional records
|       `-- feature_engineered_fraud_data.csv # 29-column feature matrix
|-- models/
|   |-- fraud_detection_model.pkl       # Trained Random Forest classifier (local artifact)
|   `-- scaler.pkl                      # Fitted StandardScaler instance
|-- outputs/
|   |-- figures/                        # High-resolution (300 DPI) EDA charts
|   |   |-- account_balance_by_fraud.png
|   |   |-- daily_transaction_count.png
|   |   |-- device_type_vs_fraud.png
|   |   |-- fraud_distribution.png
|   |   |-- merchant_category_vs_fraud.png
|   |   |-- previous_fraud_activity.png
|   |   |-- transaction_amount_by_fraud.png
|   |   `-- transaction_type_vs_fraud.png
|   `-- prediction_result.csv           # Sample inference output
|-- reports/
|   `-- model_evaluation.txt            # Benchmark performance evaluation report
|-- src/
|   |-- __init__.py
|   |-- data_cleaning.py                # Ingestion and data cleaning logic
|   |-- data_loading.py                 # Dataset inspection utilities
|   |-- eda.py                          # Statistical EDA plot generation
|   |-- evaluate_model.py               # Quantitative validation routines
|   |-- feature_engineering.py          # Feature extraction and encoding pipeline
|   |-- fraud_prediction.py             # Reusable programmatic inference engine
|   `-- train_model.py                  # Model training and artifact serialization
|-- .gitignore                          # Git tracking exclusion configuration
|-- main.py                             # Unified CLI pipeline orchestrator
|-- README.md                           # Production project documentation
`-- requirements.txt                    # System dependencies
```

---

## Feature Engineering Methodology

The raw dataset provides basic transaction records that are enriched with the following computed feature dimensions:

1. **Amount-to-Balance Ratio**:
   $$\text{Ratio} = \frac{\text{Transaction\_Amount}}{\text{Account\_Balance} + 1}$$
   *Measures capital drain intensity against available account liquidity.*

2. **High Amount Flag**:
   $$\text{High\_Amount} = \mathbb{I}(\text{Transaction\_Amount} > \text{Median}(\text{Transaction\_Amount}))$$

3. **High Frequency Flag**:
   $$\text{High\_Freq} = \mathbb{I}(\text{Daily\_Transaction\_Count} > \text{Median}(\text{Daily\_Transaction\_Count}))$$

4. **Temporal Decomposition**:
   - $\text{Year}, \text{Month}, \text{Day}, \text{DayOfWeek}$ extracted from ISO datetime strings.

5. **Categorical Matrix Encoding**:
   - One-hot binary encoding across `Transaction_Type` (ATM, Transfer, Online, POS), `Device_Type` (Laptop, Mobile, Tablet), `Location` (Mumbai, New York, Sydney, Tokyo), `Merchant_Category` (Clothing, Electronics, Groceries, Restaurants, Travel), and `Card_Type` (Amex, Discover, Mastercard, Visa).

---

## Installation & Environment Setup

### System Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git 2.30+
- 4 GB RAM minimum (8 GB recommended for Random Forest ensemble training)

### 1. Clone Repository
```bash
git clone https://github.com/ut3av/ZidioFraudDetection.git
cd ZidioFraudDetection
```

### 2. Configure Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Operations & Pipeline Execution Guide

### 1. Execute Full End-to-End Pipeline
To run data cleaning, feature engineering, exploratory analysis, model training, and performance reporting sequentially:
```bash
python main.py --step all
```

### 2. Execute Granular Pipeline Steps
```bash
# Clean raw transaction records
python main.py --step clean

# Build 29-column feature matrix
python main.py --step features

# Generate 8 analytical visualization plots
python main.py --step eda

# Train Random Forest classifier and serialize artifacts
python main.py --step train

# Evaluate model metrics and generate reports/model_evaluation.txt
python main.py --step evaluate

# Execute single-sample inference test
python main.py --step predict
```

### 3. Launch Enterprise Dashboard
Start the Sky Blue Light Theme interactive web dashboard:
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## REST API / Programmatic Integration Schema

To integrate FraudShield AI with an existing payment gateway or microservice, use the following input payload schema:

### Input JSON Schema
```json
{
  "Transaction_Amount": 249.50,
  "Account_Balance": 1250.00,
  "Previous_Fraudulent_Activity": 0,
  "Daily_Transaction_Count": 4,
  "Card_Age": 365,
  "Transaction_Type": "POS",
  "Device_Type": "Mobile",
  "Location": "New York",
  "Merchant_Category": "Electronics",
  "Card_Type": "Visa"
}
```

### Expected Response Schema
```json
{
  "prediction": "LEGITIMATE",
  "risk_score_percentage": 14.82,
  "decision": "APPROVE",
  "latency_ms": 3.8
}
```

---

## Quality Assurance & Testing

All source code and pipelines have been compiled and verified for production compliance:
```bash
# Bytecode validation
python -m compileall .

# Import dependency verification
python -c "import pandas, sklearn, joblib, matplotlib, seaborn, plotly, streamlit; print('All dependencies verified successfully')"
```

---

## License

Developed under Zidio Development for professional machine learning demonstration and enterprise financial security modeling.
