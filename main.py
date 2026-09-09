"""
Zidio Financial Fraud Detection Pipeline
Master entry point to execute and orchestrate the full machine learning workflow:
1. Data Cleaning
2. Feature Engineering
3. Exploratory Data Analysis (Visualizations)
4. Model Training & Evaluation
5. Model Evaluation Report Generation
6. Sample Inference Verification
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.eda import run_eda
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.fraud_prediction import predict_sample


def run_full_pipeline(skip_eda: bool = False) -> None:
    print("==========================================================")
    print("  FINANCIAL FRAUD DETECTION PIPELINE - ZIDIO DEVELOPMENT  ")
    print("==========================================================")

    # Step 1: Data Cleaning
    print("\n[STEP 1/5] Executing Data Cleaning...")
    clean_data()

    # Step 2: Feature Engineering
    print("\n[STEP 2/5] Executing Feature Engineering...")
    engineer_features()

    # Step 3: Exploratory Data Analysis
    if not skip_eda:
        print("\n[STEP 3/5] Generating EDA Visualizations...")
        run_eda()
    else:
        print("\n[STEP 3/5] Skipping EDA Visualizations (--skip-eda passed).")

    # Step 4: Model Training
    print("\n[STEP 4/5] Training Random Forest Classifier...")
    train_model()

    # Step 5: Evaluation and Inference
    print("\n[STEP 5/5] Generating Evaluation Report & Testing Inference...")
    evaluate_model()
    predict_sample()

    print("\n==========================================================")
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!              ")
    print("==========================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end Financial Fraud Detection Machine Learning Pipeline."
    )
    parser.add_argument(
        "--step",
        choices=["all", "clean", "features", "eda", "train", "evaluate", "predict"],
        default="all",
        help="Specify the pipeline step to execute (default: all)"
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA generation when running all steps"
    )

    args = parser.parse_args()

    if args.step == "all":
        run_full_pipeline(skip_eda=args.skip_eda)
    elif args.step == "clean":
        clean_data()
    elif args.step == "features":
        engineer_features()
    elif args.step == "eda":
        run_eda()
    elif args.step == "train":
        train_model()
    elif args.step == "evaluate":
        evaluate_model()
    elif args.step == "predict":
        predict_sample()


if __name__ == "__main__":
    main()
