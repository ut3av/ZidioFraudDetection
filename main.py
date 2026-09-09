"""
Zidio Financial Fraud Detection Pipeline & Entry Point
Master entry point supporting dual execution modes:
1. Streamlit Cloud / Web App Execution (Automatically renders the dashboard when launched via Streamlit)
2. CLI Machine Learning Pipeline Orchestration (Data cleaning, feature engineering, EDA, training, evaluation, inference)
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    import streamlit as st
except ImportError:
    st = None


def is_streamlit_running() -> bool:
    """Detect if executed inside a Streamlit runner (e.g. Streamlit Cloud)."""
    if st is not None and hasattr(st, "runtime") and hasattr(st.runtime, "exists"):
        return st.runtime.exists()
    return False


if is_streamlit_running():
    # Streamlit Cloud executes main.py directly when configured as the main module.
    # Seamlessly forward execution to the enterprise dashboard.
    import runpy
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    runpy.run_path(str(dashboard_path), run_name="__main__")

else:
    # Standard CLI Pipeline Runner
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
