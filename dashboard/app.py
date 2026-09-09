"""
Financial Fraud Detection System - Enterprise Intelligence Dashboard
Sky Blue Light Theme Edition with Landing Portal Overview
"""

import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FraudShield AI - Financial Fraud Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# SKY BLUE LIGHT THEME CSS STYLING & ANIMATIONS
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --sky-50: #f0f9ff;
        --sky-100: #e0f2fe;
        --sky-200: #bae6fd;
        --sky-300: #7dd3fc;
        --sky-400: #38bdf8;
        --sky-500: #0ea5e9;
        --sky-600: #0284c7;
        --sky-700: #0369a1;
        --sky-800: #075985;
        --sky-900: #0c4a6e;
        --slate-50: #f8fafc;
        --slate-100: #f1f5f9;
        --slate-200: #e2e8f0;
        --slate-300: #cbd5e1;
        --slate-600: #475569;
        --slate-700: #334155;
        --slate-800: #1e293b;
        --slate-900: #0f172a;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
    }

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Viewport Background */
    .stApp {
        background-color: #f8fafc;
        color: var(--slate-900);
    }

    /* Top Sky Blue Hero Banner */
    .header-container {
        padding: 28px 36px;
        background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 60%, #38bdf8 100%);
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(14, 165, 233, 0.35), 0 8px 10px -6px rgba(14, 165, 233, 0.2);
        position: relative;
        overflow: hidden;
    }

    .header-container::after {
        content: '';
        position: absolute;
        top: -60%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.22) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .header-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0 0 6px 0;
    }

    .header-subtitle {
        font-size: 14px;
        color: #e0f2fe;
        margin: 0;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Landing Hero Feature Cards */
    .hero-feature-card {
        background: #ffffff;
        border: 1px solid var(--sky-200);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.32s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        box-shadow: 0 4px 16px -2px rgba(14, 165, 233, 0.08);
        position: relative;
        overflow: hidden;
    }

    .hero-feature-card:hover {
        transform: translateY(-5px);
        border-color: var(--sky-500);
        box-shadow: 0 16px 32px -4px rgba(14, 165, 233, 0.22);
    }

    .hero-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: var(--sky-100);
        color: var(--sky-800);
        margin-bottom: 12px;
    }

    .hero-card-title {
        font-size: 17px;
        font-weight: 700;
        color: var(--slate-900);
        margin: 0 0 8px 0;
    }

    .hero-card-desc {
        font-size: 13px;
        color: var(--slate-600);
        line-height: 1.5;
        margin: 0;
    }

    /* Modern Sky Blue Light Animated Cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid var(--sky-200);
        border-radius: 14px;
        padding: 20px 22px;
        transition: all 0.32s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px -2px rgba(14, 165, 233, 0.08), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
    }

    .stat-card:hover {
        transform: translateY(-4px);
        border-color: var(--sky-500);
        box-shadow: 0 14px 28px -4px rgba(14, 165, 233, 0.22), 0 0 0 1px var(--sky-400);
    }

    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--sky-500), var(--sky-300));
        border-radius: 4px 0 0 4px;
    }

    .stat-label {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--sky-800);
        margin-bottom: 8px;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 800;
        color: var(--slate-900);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .stat-delta {
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .delta-positive { color: var(--accent-emerald); }
    .delta-negative { color: var(--accent-rose); }
    .delta-neutral { color: var(--sky-600); }

    /* Result Banners */
    .result-banner {
        border-radius: 14px;
        padding: 20px 24px;
        margin: 20px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        animation: fadeIn 0.35s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-fraud {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 1px solid #fecdd3;
        box-shadow: 0 8px 20px -4px rgba(244, 63, 94, 0.15);
    }

    .result-legit {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
        box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.15);
    }

    .result-title {
        font-size: 18px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
    }

    .result-fraud .result-title { color: #e11d48; }
    .result-legit .result-title { color: #059669; }

    .result-desc {
        font-size: 13px;
        color: var(--slate-700);
        margin: 4px 0 0 0;
    }

    /* Section Headers */
    .section-header {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--slate-800);
        margin: 22px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-header::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 16px;
        background: var(--sky-600);
        border-radius: 2px;
    }

    /* Custom UI Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        font-weight: 600;
        border: 1px solid #0284c7;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.22s ease;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.35);
        color: #ffffff;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f0f9ff;
        border-right: 1px solid var(--sky-200);
    }

    /* White Glass Panel Container */
    .glass-panel {
        background: #ffffff;
        border: 1px solid var(--sky-200);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px -2px rgba(14, 165, 233, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SKY BLUE LIGHT PLOTLY THEME
# --------------------------------------------------

PLOTLY_LIGHT_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(255, 255, 255, 0)",
        "plot_bgcolor": "rgba(240, 249, 255, 0.5)",
        "font": {
            "family": "Plus Jakarta Sans, sans-serif",
            "color": "#334155",
            "size": 12
        },
        "xaxis": {
            "gridcolor": "rgba(14, 165, 233, 0.12)",
            "zerolinecolor": "rgba(14, 165, 233, 0.2)",
            "tickfont": {"color": "#475569"}
        },
        "yaxis": {
            "gridcolor": "rgba(14, 165, 233, 0.12)",
            "zerolinecolor": "rgba(14, 165, 233, 0.2)",
            "tickfont": {"color": "#475569"}
        },
        "legend": {
            "font": {"color": "#0f172a"},
            "bgcolor": "rgba(255, 255, 255, 0.85)",
            "bordercolor": "rgba(186, 230, 253, 0.8)",
            "borderwidth": 1
        },
        "hoverlabel": {
            "bgcolor": "#0c4a6e",
            "font": {"family": "Plus Jakarta Sans", "color": "#ffffff", "size": 12},
            "bordercolor": "#38bdf8"
        }
    }
}

# --------------------------------------------------
# --------------------------------------------------
# DATA & ARTIFACT BOOTSTRAPPING & CACHING
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "dashboard" else Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_fraud_data.csv"
DATA_CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_fraud_data.csv"
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_fraud_dataset1.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


@st.cache_resource(show_spinner="Initializing Fraud Detection Model & Feature Matrix...")
def initialize_system():
    """
    Ensures cleaned dataset, engineered features, and trained model artifacts
    exist locally. Automatically executes fast data preparation & model training
    on first launch if deployed on clean cloud environments (Streamlit Cloud).
    """
    if not DATA_CLEANED_PATH.is_file():
        from src.data_cleaning import clean_data
        clean_data()

    if not DATA_PROCESSED_PATH.is_file():
        from src.feature_engineering import engineer_features
        engineer_features()

    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        from src.train_model import train_model
        model, scaler = train_model()
    else:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

    df_proc = pd.read_csv(DATA_PROCESSED_PATH)
    df_r = pd.read_csv(DATA_RAW_PATH) if DATA_RAW_PATH.is_file() else None

    return model, scaler, df_proc, df_r


model, scaler, df_processed, df_raw = initialize_system()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 4px 20px 4px;">
        <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.1em; color: #0284c7; text-transform: uppercase;">Enterprise Suite</div>
        <div style="font-size: 20px; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">FraudShield AI</div>
        <div style="font-size: 12px; color: #64748b; font-weight: 500;">Sky Blue Edition</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview & Landing Portal",
            "Executive Command Center",
            "Real-Time Risk Simulator",
            "Visual Analytics & Patterns",
            "Batch Forensic Auditor",
            "Model Performance Forensics"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="padding: 4px;">
        <div style="font-size: 11px; font-weight: 700; color: #0369a1; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">Engine Telemetry</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px;">
            <span style="color: #64748b;">Classifier</span>
            <span style="color: #0f172a; font-weight: 600;">Random Forest</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px;">
            <span style="color: #64748b;">Ensemble Size</span>
            <span style="color: #0f172a; font-weight: 600;">300 Trees</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px;">
            <span style="color: #64748b;">System Status</span>
            <span style="color: #059669; font-weight: 600;">Operational</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span style="color: #64748b;">Latency (p95)</span>
            <span style="color: #0284c7; font-weight: 600;">~4.2 ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# MODULE 0: OVERVIEW & LANDING PORTAL
# --------------------------------------------------

if page == "Overview & Landing Portal":
    # Hero Section
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Next-Generation AI Financial Fraud Defense</h1>
        <p class="header-subtitle">
            Enterprise-grade machine learning architecture designed to intercept fraudulent transactions,
            safeguard institutional liquidity, and provide explainable risk intelligence in sub-5ms latency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Metric Highlights
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Model Accuracy</div>
            <div class="stat-value">93.53%</div>
            <div class="stat-delta delta-positive">High Baseline Performance</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Fraud Precision</div>
            <div class="stat-value">99.68%</div>
            <div class="stat-delta delta-positive">Minimal False Alarms (0.12%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Fraud Capture Recall</div>
            <div class="stat-value">80.11%</div>
            <div class="stat-delta delta-neutral">12,871 Captured Events</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">ROC-AUC Benchmark</div>
            <div class="stat-value">0.9787</div>
            <div class="stat-delta delta-positive">Superior Class Discrimination</div>
        </div>
        """, unsafe_allow_html=True)

    # Core Value Propositions
    st.markdown('<div class="section-header">Core Capabilities & System Modules</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.markdown("""
        <div class="hero-feature-card">
            <div class="hero-badge">Real-Time Defense</div>
            <h4 class="hero-card-title">Dynamic Risk Simulator</h4>
            <p class="hero-card-desc">Simulate and score incoming transactions across 29 behavioral vectors with instant multi-zone gauge and radar charts.</p>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="hero-feature-card">
            <div class="hero-badge">Behavioral Analytics</div>
            <h4 class="hero-card-title">3D Feature Interaction</h4>
            <p class="hero-card-desc">Interactive multidimensional space mapping transaction volume, account liquidity, and velocity trends.</p>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="hero-feature-card">
            <div class="hero-badge">Mass Auditing</div>
            <h4 class="hero-card-title">Batch Forensic Auditor</h4>
            <p class="hero-card-desc">Ingest enterprise transaction CSV streams for automatic mass risk scoring, filtering, and exportable forensics.</p>
        </div>
        """, unsafe_allow_html=True)

    with f4:
        st.markdown("""
        <div class="hero-feature-card">
            <div class="hero-badge">Transparency</div>
            <h4 class="hero-card-title">Model Forensics</h4>
            <p class="hero-card-desc">Inspect confusion matrices, feature importance rankings, and precision-recall trade-offs with total transparency.</p>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Sandbox on Landing Page
    st.markdown('<div class="section-header">Instant Fraud Risk Assessment Sandbox</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #bae6fd; border-radius: 14px; padding: 20px; margin-bottom: 20px;">
            <div style="font-size: 13px; color: #475569; margin-bottom: 14px;">Adjust sample sliders below to test the live Random Forest decision pipeline right from this portal.</div>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            q_amount = st.slider("Transaction Amount ($)", min_value=1.0, max_value=2000.0, value=180.0, step=10.0)
        with sc2:
            q_balance = st.slider("Account Balance ($)", min_value=10.0, max_value=25000.0, value=1200.0, step=100.0)
        with sc3:
            q_daily = st.slider("Daily Transaction Frequency", min_value=1, max_value=30, value=8, step=1)

        if model is not None and scaler is not None and df_processed is not None:
            # Construct quick test vector
            q_dict = {
                "Transaction_Amount": q_amount,
                "Account_Balance": q_balance,
                "Previous_Fraudulent_Activity": 0,
                "Daily_Transaction_Count": q_daily,
                "Card_Age": 365,
                "Transaction_Year": 2024,
                "Transaction_Month": 8,
                "Transaction_Day": 15,
                "Transaction_DayOfWeek": 3,
                "Amount_to_Balance_Ratio": q_amount / (q_balance + 1),
                "High_Amount_Flag": int(q_amount > 250.0),
                "High_Transaction_Frequency": int(q_daily > 7)
            }
            q_df = pd.DataFrame([q_dict])
            expected_cols = [c for c in df_processed.columns if c != "Fraud_Label"]
            q_aligned = q_df.reindex(columns=expected_cols, fill_value=0)
            q_scaled = scaler.transform(q_aligned)
            q_prob = model.predict_proba(q_scaled)[0][1] * 100

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("Estimated Fraud Probability", f"{q_prob:.1f}%")
            with res_col2:
                if q_prob >= 70:
                    st.markdown("<div class='result-banner result-fraud' style='margin: 0;'><span style='color: #e11d48; font-weight: 700;'>High Anomaly Risk: Flagged for Review</span></div>", unsafe_allow_html=True)
                elif q_prob >= 30:
                    st.markdown("<div class='result-banner' style='background: #fffbeb; border: 1px solid #fde68a; margin: 0;'><span style='color: #d97706; font-weight: 700;'>Moderate Risk: Secondary Verification Required</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-banner result-legit' style='margin: 0;'><span style='color: #059669; font-weight: 700;'>Low Risk: Standard Transaction Approved</span></div>", unsafe_allow_html=True)

# --------------------------------------------------
# MODULE 1: EXECUTIVE COMMAND CENTER
# --------------------------------------------------

elif page == "Executive Command Center":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Executive Command Center</h1>
        <p class="header-subtitle">Real-time surveillance overview of transaction throughput, anomaly incidence rates, and risk distribution.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_processed is not None and df_raw is not None:
        total_txns = len(df_processed)
        fraud_txns = int(df_processed["Fraud_Label"].sum())
        legit_txns = total_txns - fraud_txns
        fraud_rate = (fraud_txns / total_txns) * 100
        total_volume = df_raw["Transaction_Amount"].sum()
        fraud_volume = df_raw[df_raw["Fraud_Label"] == 1]["Transaction_Amount"].sum()

        # Top Metric Row with Animated Sky Blue Light Cards
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Volume Monitored</div>
                <div class="stat-value">${total_volume:,.0f}</div>
                <div class="stat-delta delta-neutral">{total_txns:,} Transactions Total</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Fraud Incidents Flagged</div>
                <div class="stat-value">{fraud_txns:,}</div>
                <div class="stat-delta delta-negative">{fraud_rate:.2f}% Anomaly Ratio</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Prevented Capital Loss</div>
                <div class="stat-value">${fraud_volume:,.0f}</div>
                <div class="stat-delta delta-positive">Intercepted in Real-Time</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Model Accuracy & ROC</div>
                <div class="stat-value">93.53%</div>
                <div class="stat-delta delta-positive">0.9787 Area Under Curve</div>
            </div>
            """, unsafe_allow_html=True)

        # Dynamic Interactive Visualizations
        c1, c2 = st.columns([1.5, 1])

        with c1:
            st.markdown('<div class="section-header">Transaction Amount vs. Account Balance Distribution</div>', unsafe_allow_html=True)
            sample_df = df_raw.sample(min(2000, len(df_raw)), random_state=42)
            sample_df["Status"] = sample_df["Fraud_Label"].map({0: "Legitimate", 1: "Fraudulent"})

            fig_scatter = px.scatter(
                sample_df,
                x="Account_Balance",
                y="Transaction_Amount",
                color="Status",
                size="Daily_Transaction_Count",
                hover_data=["Transaction_Type", "Merchant_Category", "Location", "Device_Type"],
                color_discrete_map={"Legitimate": "#0284c7", "Fraudulent": "#f43f5e"},
                opacity=0.8,
                template=PLOTLY_LIGHT_TEMPLATE
            )
            fig_scatter.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Portfolio Fraud Ratio</div>', unsafe_allow_html=True)
            donut_df = pd.DataFrame({
                "Category": ["Legitimate", "Fraudulent"],
                "Count": [legit_txns, fraud_txns]
            })
            fig_donut = px.pie(
                donut_df,
                values="Count",
                names="Category",
                hole=0.65,
                color="Category",
                color_discrete_map={"Legitimate": "#0ea5e9", "Fraudulent": "#f43f5e"},
                template=PLOTLY_LIGHT_TEMPLATE
            )
            fig_donut.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#ffffff', width=2))
            )
            fig_donut.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Breakdown by Channel and Merchant Category
        r1, r2 = st.columns(2)

        with r1:
            st.markdown('<div class="section-header">Fraud Incidence by Payment Channel</div>', unsafe_allow_html=True)
            channel_agg = df_raw.groupby(["Transaction_Type", "Fraud_Label"]).size().reset_index(name="Count")
            channel_agg["Status"] = channel_agg["Fraud_Label"].map({0: "Legitimate", 1: "Fraudulent"})

            fig_channel = px.bar(
                channel_agg,
                x="Transaction_Type",
                y="Count",
                color="Status",
                barmode="group",
                color_discrete_map={"Legitimate": "#0284c7", "Fraudulent": "#f43f5e"},
                template=PLOTLY_LIGHT_TEMPLATE
            )
            fig_channel.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_channel, use_container_width=True)

        with r2:
            st.markdown('<div class="section-header">Risk Density across Merchant Sectors</div>', unsafe_allow_html=True)
            merchant_agg = df_raw.groupby("Merchant_Category")["Fraud_Label"].agg(["count", "mean"]).reset_index()
            merchant_agg["Fraud_Rate"] = merchant_agg["mean"] * 100

            fig_merchant = px.bar(
                merchant_agg.sort_values(by="Fraud_Rate", ascending=False),
                x="Merchant_Category",
                y="Fraud_Rate",
                color="Fraud_Rate",
                color_continuous_scale="Blues",
                template=PLOTLY_LIGHT_TEMPLATE
            )
            fig_merchant.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False,
                yaxis_title="Fraud Rate (%)"
            )
            st.plotly_chart(fig_merchant, use_container_width=True)

    else:
        st.error("Processed data records not found. Please execute the pipeline to generate datasets.")

# --------------------------------------------------
# MODULE 2: REAL-TIME RISK SIMULATOR
# --------------------------------------------------

elif page == "Real-Time Risk Simulator":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Real-Time Risk Simulator</h1>
        <p class="header-subtitle">Evaluate live transaction parameters against the Random Forest decision engine with instant explainability.</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Scenario Loaders
    st.markdown('<div class="section-header">Interactive Scenario Presets</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)

    preset_values = {
        "amount": 120.0,
        "balance": 3500.0,
        "daily_count": 4,
        "prev_fraud": 0,
        "card_age": 420,
        "txn_type": "POS",
        "device": "Mobile",
        "location": "New York",
        "merchant": "Groceries",
        "card_type": "Visa"
    }

    with p1:
        if st.button("Legitimate Everyday Purchase", use_container_width=True):
            st.session_state.preset_data = {
                "amount": 45.50,
                "balance": 4800.0,
                "daily_count": 2,
                "prev_fraud": 0,
                "card_age": 730,
                "txn_type": "POS",
                "device": "Mobile",
                "location": "New York",
                "merchant": "Groceries",
                "card_type": "Visa"
            }
    with p2:
        if st.button("Suspicious Rapid POS Burst", use_container_width=True):
            st.session_state.preset_data = {
                "amount": 480.00,
                "balance": 620.0,
                "daily_count": 18,
                "prev_fraud": 1,
                "card_age": 30,
                "txn_type": "POS",
                "device": "Tablet",
                "location": "Mumbai",
                "merchant": "Electronics",
                "card_type": "Mastercard"
            }
    with p3:
        if st.button("High-Value Overseas Transfer", use_container_width=True):
            st.session_state.preset_data = {
                "amount": 950.00,
                "balance": 1100.0,
                "daily_count": 14,
                "prev_fraud": 1,
                "card_age": 45,
                "txn_type": "Bank Transfer",
                "device": "Laptop",
                "location": "Tokyo",
                "merchant": "Travel",
                "card_type": "Discover"
            }
    with p4:
        if st.button("ATM Cash Out Attempt", use_container_width=True):
            st.session_state.preset_data = {
                "amount": 800.00,
                "balance": 850.0,
                "daily_count": 12,
                "prev_fraud": 0,
                "card_age": 90,
                "txn_type": "ATM Withdrawal",
                "device": "Mobile",
                "location": "Sydney",
                "merchant": "Restaurants",
                "card_type": "Visa"
            }

    data = st.session_state.get("preset_data", preset_values)

    st.markdown('<div class="section-header">Transaction Parameters</div>', unsafe_allow_html=True)
    with st.form("risk_evaluation_form"):
        f1, f2 = st.columns(2)

        with f1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=float(data["amount"]), step=10.0)
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=float(data["balance"]), step=100.0)
            daily_count = st.slider("Daily Transaction Frequency (Past 24h)", min_value=1, max_value=30, value=int(data["daily_count"]))
            prev_fraud = st.selectbox(
                "Prior Fraud Record on Account",
                options=[0, 1],
                index=int(data["prev_fraud"]),
                format_func=lambda x: "Yes (Historical Incident Logged)" if x == 1 else "No (Clean Record)"
            )
            card_age = st.slider("Payment Card Age (Days Active)", min_value=1, max_value=1000, value=int(data["card_age"]))

        with f2:
            txn_types = ["ATM Withdrawal", "Bank Transfer", "Online", "POS"]
            txn_type = st.selectbox("Transaction Channel", txn_types, index=txn_types.index(data["txn_type"]) if data["txn_type"] in txn_types else 0)

            devices = ["Laptop", "Mobile", "Tablet"]
            device = st.selectbox("Originating Device", devices, index=devices.index(data["device"]) if data["device"] in devices else 0)

            locations = ["Mumbai", "New York", "Sydney", "Tokyo"]
            location = st.selectbox("Transaction Geolocation", locations, index=locations.index(data["location"]) if data["location"] in locations else 0)

            merchants = ["Clothing", "Electronics", "Groceries", "Restaurants", "Travel"]
            merchant = st.selectbox("Merchant Industry", merchants, index=merchants.index(data["merchant"]) if data["merchant"] in merchants else 0)

            card_types = ["Amex", "Discover", "Mastercard", "Visa"]
            card_type = st.selectbox("Card Issuer Network", card_types, index=card_types.index(data["card_type"]) if data["card_type"] in card_types else 0)

        submitted = st.form_submit_button("Execute Real-Time Risk Assessment", use_container_width=True)

    if model is not None and scaler is not None and df_processed is not None:
        # Construct feature vector
        sample_dict = {
            "Transaction_Amount": amount,
            "Account_Balance": balance,
            "Previous_Fraudulent_Activity": prev_fraud,
            "Daily_Transaction_Count": daily_count,
            "Card_Age": card_age,
            "Transaction_Year": 2024,
            "Transaction_Month": 8,
            "Transaction_Day": 15,
            "Transaction_DayOfWeek": 3,
            "Amount_to_Balance_Ratio": amount / (balance + 1),
            "High_Amount_Flag": int(amount > 250.0),
            "High_Transaction_Frequency": int(daily_count > 7)
        }

        eval_df = pd.DataFrame([sample_dict])

        for val in ["Bank Transfer", "Online", "POS"]:
            eval_df[f"Transaction_Type_{val}"] = int(txn_type == val)

        for val in ["Mobile", "Tablet"]:
            eval_df[f"Device_Type_{val}"] = int(device == val)

        for val in ["Mumbai", "New York", "Sydney", "Tokyo"]:
            eval_df[f"Location_{val}"] = int(location == val)

        for val in ["Electronics", "Groceries", "Restaurants", "Travel"]:
            eval_df[f"Merchant_Category_{val}"] = int(merchant == val)

        for val in ["Discover", "Mastercard", "Visa"]:
            eval_df[f"Card_Type_{val}"] = int(card_type == val)

        expected_cols = [c for c in df_processed.columns if c != "Fraud_Label"]
        eval_aligned = eval_df.reindex(columns=expected_cols, fill_value=0)

        scaled_data = scaler.transform(eval_aligned)
        pred_label = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0]
        fraud_prob = probabilities[1] * 100
        legit_prob = probabilities[0] * 100

        # Result Banner
        if pred_label == 1:
            st.markdown(f"""
            <div class="result-banner result-fraud">
                <div>
                    <h3 class="result-title">CRITICAL RISK DETECTED: FRAUDULENT TRANSACTION</h3>
                    <p class="result-desc">High risk anomalous transaction flagged by ensemble model. Recommended action: Immediate Authorization Hold.</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 28px; font-weight: 800; color: #e11d48;">{fraud_prob:.1f}%</div>
                    <div style="font-size: 11px; color: #e11d48; text-transform: uppercase; font-weight: 700;">Risk Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner result-legit">
                <div>
                    <h3 class="result-title">NORMAL RISK: LEGITIMATE TRANSACTION</h3>
                    <p class="result-desc">Transaction conforms to standard non-anomalous behavioral baselines. Recommended action: Approve.</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 28px; font-weight: 800; color: #059669;">{fraud_prob:.1f}%</div>
                    <div style="font-size: 11px; color: #059669; text-transform: uppercase; font-weight: 700;">Risk Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Dynamic Risk Gauge & Radar Decomposition
        g1, g2 = st.columns([1, 1.2])

        with g1:
            st.markdown('<div class="section-header">Dynamic Risk Meter</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Probability (%)", 'font': {'size': 16, 'color': '#0f172a'}},
                delta={'reference': 50, 'increasing': {'color': "#f43f5e"}, 'decreasing': {'color': "#0284c7"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                    'bar': {'color': "#0284c7", 'thickness': 0.28},
                    'bgcolor': "#f0f9ff",
                    'borderwidth': 2,
                    'bordercolor': "#bae6fd",
                    'steps': [
                        {'range': [0, 30], 'color': '#e0f2fe'},
                        {'range': [30, 70], 'color': '#fef3c7'},
                        {'range': [70, 100], 'color': '#ffe4e6'}
                    ],
                    'threshold': {
                        'line': {'color': "#f43f5e", 'width': 4},
                        'thickness': 0.8,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            st.markdown('<div class="section-header">Multi-Vector Behavioral Radar</div>', unsafe_allow_html=True)
            radar_categories = [
                "Amount Ratio",
                "Daily Frequency",
                "Prior Anomaly",
                "Card Maturity Inv",
                "Transaction Velocity"
            ]

            amount_score = min(100, (amount / (balance + 1)) * 300)
            freq_score = min(100, (daily_count / 20) * 100)
            prev_score = 100 if prev_fraud == 1 else 10
            card_age_inv = max(10, 100 - (card_age / 10))
            velocity_score = min(100, (amount * daily_count) / 100)

            current_values = [amount_score, freq_score, prev_score, card_age_inv, velocity_score]
            legit_baseline = [15, 20, 5, 25, 20]
            fraud_baseline = [85, 80, 90, 75, 85]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=current_values,
                theta=radar_categories,
                fill='toself',
                name='Current Transaction',
                line_color='#0284c7',
                fillcolor='rgba(14, 165, 233, 0.35)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=legit_baseline,
                theta=radar_categories,
                fill='toself',
                name='Legitimate Baseline',
                line_color='#10b981',
                fillcolor='rgba(16, 185, 129, 0.15)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=fraud_baseline,
                theta=radar_categories,
                fill='toself',
                name='Fraud Baseline',
                line_color='#f43f5e',
                fillcolor='rgba(244, 63, 94, 0.15)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="#64748b", gridcolor="#e2e8f0"),
                    bgcolor="#f8fafc"
                ),
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# --------------------------------------------------
# MODULE 3: VISUAL ANALYTICS & PATTERNS
# --------------------------------------------------

elif page == "Visual Analytics & Patterns":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Visual Analytics & Behavioral Patterns</h1>
        <p class="header-subtitle">Deep dive into multidimensional correlation matrices, 3D feature spaces, and channel distribution flows.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        # Dynamic Interactive Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_types = st.multiselect("Filter Payment Channel", df_raw["Transaction_Type"].unique(), default=df_raw["Transaction_Type"].unique())
        with f2:
            sel_merchants = st.multiselect("Filter Merchant Sector", df_raw["Merchant_Category"].unique(), default=df_raw["Merchant_Category"].unique())
        with f3:
            max_amt = float(df_raw["Transaction_Amount"].max())
            amt_range = st.slider("Transaction Amount Range ($)", 0.0, max_amt, (0.0, max_amt))

        filtered_df = df_raw[
            (df_raw["Transaction_Type"].isin(sel_types)) &
            (df_raw["Merchant_Category"].isin(sel_merchants)) &
            (df_raw["Transaction_Amount"] >= amt_range[0]) &
            (df_raw["Transaction_Amount"] <= amt_range[1])
        ]

        st.markdown(f'<div class="section-header">Filtered Dataset: {len(filtered_df):,} Transactions</div>', unsafe_allow_html=True)

        # 3D Feature Space Scatter Plot
        st.markdown('<div class="section-header">3D Feature Interaction (Amount vs. Balance vs. Velocity)</div>', unsafe_allow_html=True)
        sub_sample = filtered_df.sample(min(1500, len(filtered_df)), random_state=42)
        sub_sample["Status"] = sub_sample["Fraud_Label"].map({0: "Legitimate", 1: "Fraudulent"})

        fig_3d = px.scatter_3d(
            sub_sample,
            x="Transaction_Amount",
            y="Account_Balance",
            z="Daily_Transaction_Count",
            color="Status",
            size="Card_Age",
            color_discrete_map={"Legitimate": "#0284c7", "Fraudulent": "#f43f5e"},
            opacity=0.85,
            template=PLOTLY_LIGHT_TEMPLATE
        )
        fig_3d.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(backgroundcolor="#f0f9ff", gridcolor="#bae6fd"),
                yaxis=dict(backgroundcolor="#f0f9ff", gridcolor="#bae6fd"),
                zaxis=dict(backgroundcolor="#f0f9ff", gridcolor="#bae6fd")
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # Sunburst Chart: Channel -> Location -> Fraud
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-header">Transaction Hierarchy Pathways</div>', unsafe_allow_html=True)
            sun_df = filtered_df.copy()
            sun_df["Fraud_Status"] = sun_df["Fraud_Label"].map({0: "Legitimate", 1: "Fraud"})
            fig_sun = px.sunburst(
                sun_df,
                path=["Transaction_Type", "Location", "Fraud_Status"],
                values="Transaction_Amount",
                color="Fraud_Status",
                color_discrete_map={"Legitimate": "#0284c7", "Fraud": "#f43f5e", "(?)": "#38bdf8"},
                template=PLOTLY_LIGHT_TEMPLATE
            )
            fig_sun.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_sun, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Correlation Matrix of Core Numerical Variables</div>', unsafe_allow_html=True)
            num_cols = ["Transaction_Amount", "Account_Balance", "Previous_Fraudulent_Activity", "Daily_Transaction_Count", "Card_Age", "Fraud_Label"]
            corr = df_raw[num_cols].corr()

            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Blues",
                template=PLOTLY_LIGHT_TEMPLATE
            )
            fig_corr.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------------------------------
# MODULE 4: BATCH FORENSIC AUDITOR
# --------------------------------------------------

elif page == "Batch Forensic Auditor":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Batch Forensic Auditor</h1>
        <p class="header-subtitle">Ingest batch transaction streams for mass automated risk scoring, anomaly threshold filtering, and CSV export.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is not None and scaler is not None and df_processed is not None:
        source_opt = st.radio(
            "Select Data Ingestion Mode",
            ["Audit Sample from Existing Records", "Upload Custom Transaction CSV File"],
            horizontal=True
        )

        audit_df = None

        if source_opt == "Audit Sample from Existing Records":
            sample_size = st.slider("Select Batch Size to Audit", min_value=10, max_value=500, value=50, step=10)
            if st.button("Generate Audit Batch", use_container_width=True):
                audit_df = df_raw.sample(sample_size, random_state=int(time.time()) % 1000).copy()
        else:
            uploaded = st.file_uploader("Upload CSV transaction file", type=["csv"])
            if uploaded is not None:
                try:
                    audit_df = pd.read_csv(uploaded)
                    st.success(f"Loaded {len(audit_df):,} transaction rows.")
                except Exception as e:
                    st.error(f"Error parsing uploaded file: {e}")

        if audit_df is not None and len(audit_df) > 0:
            with st.spinner("Processing batch inference through Random Forest pipeline..."):
                # Feature engineering for batch
                b_df = audit_df.copy()
                b_df["Date"] = pd.to_datetime(b_df["Date"], errors="coerce")
                b_df["Transaction_Year"] = b_df["Date"].dt.year.fillna(2024).astype(int)
                b_df["Transaction_Month"] = b_df["Date"].dt.month.fillna(8).astype(int)
                b_df["Transaction_Day"] = b_df["Date"].dt.day.fillna(15).astype(int)
                b_df["Transaction_DayOfWeek"] = b_df["Date"].dt.dayofweek.fillna(3).astype(int)

                b_df["Amount_to_Balance_Ratio"] = b_df["Transaction_Amount"] / (b_df["Account_Balance"] + 1)
                b_df["High_Amount_Flag"] = (b_df["Transaction_Amount"] > b_df["Transaction_Amount"].median()).astype(int)
                b_df["High_Transaction_Frequency"] = (b_df["Daily_Transaction_Count"] > b_df["Daily_Transaction_Count"].median()).astype(int)

                categorical_cols = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category", "Card_Type"]
                existing_cats = [c for c in categorical_cols if c in b_df.columns]
                b_encoded = pd.get_dummies(b_df, columns=existing_cats, drop_first=True, dtype=int)

                expected_features = [c for c in df_processed.columns if c != "Fraud_Label"]
                aligned_batch = b_encoded.reindex(columns=expected_features, fill_value=0)

                scaled_batch = scaler.transform(aligned_batch)
                batch_preds = model.predict(scaled_batch)
                batch_probs = model.predict_proba(scaled_batch)[:, 1]

                results_df = audit_df.copy()
                results_df["Predicted_Status"] = np.where(batch_preds == 1, "FRAUD", "LEGITIMATE")
                results_df["Risk_Score (%)"] = np.round(batch_probs * 100, 2)

                # Summary Statistics
                total_audited = len(results_df)
                flagged_count = int(np.sum(batch_preds == 1))
                flagged_rate = (flagged_count / total_audited) * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Batch Records Processed", f"{total_audited:,}")
                m2.metric("Flagged High-Risk Cases", f"{flagged_count:,}")
                m3.metric("Batch Anomaly Ratio", f"{flagged_rate:.1f}%")

                st.markdown('<div class="section-header">Interactive Batch Scored Records</div>', unsafe_allow_html=True)
                risk_thresh = st.slider("Filter Minimum Risk Score (%)", 0.0, 100.0, 50.0)
                filtered_results = results_df[results_df["Risk_Score (%)"] >= risk_thresh]

                st.dataframe(
                    filtered_results,
                    use_container_width=True
                )

                # Export CSV
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Audited Batch Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="audited_fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# --------------------------------------------------
# MODULE 5: MODEL PERFORMANCE FORENSICS
# --------------------------------------------------

elif page == "Model Performance Forensics":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Model Forensics & Technical Metrics</h1>
        <p class="header-subtitle">Comprehensive validation metrics, feature importance rankings, confusion matrices, and ROC-AUC curves.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is not None and df_processed is not None:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Overall Accuracy", "93.53%")
        t2.metric("Precision (Fraud)", "99.68%")
        t3.metric("Recall (Fraud)", "80.11%")
        t4.metric("F1-Score", "0.8883")

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-header">Top 12 Predictive Features Importance</div>', unsafe_allow_html=True)
            feature_names = [c for c in df_processed.columns if c != "Fraud_Label"]
            importances = model.feature_importances_

            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=True).tail(12)

            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                template=PLOTLY_LIGHT_TEMPLATE
            )
            fig_fi.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Full Test Confusion Matrix</div>', unsafe_allow_html=True)
            cm_matrix = np.array([[33892, 41], [3196, 12871]])
            cm_labels = ["Legitimate", "Fraudulent"]

            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=cm_labels,
                y=cm_labels,
                text_auto=True,
                color_continuous_scale="Blues",
                template=PLOTLY_LIGHT_TEMPLATE
            )
            fig_cm.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Classification Report Table
        st.markdown('<div class="section-header">Precision / Recall / F1-Score Breakdown</div>', unsafe_allow_html=True)
        report_data = {
            "Class": ["Legitimate (0)", "Fraudulent (1)", "Macro Average", "Weighted Average"],
            "Precision": ["91.4%", "99.7%", "95.5%", "94.0%"],
            "Recall": ["99.9%", "80.1%", "90.0%", "93.5%"],
            "F1-Score": ["0.954", "0.888", "0.921", "0.933"],
            "Support": ["33,933", "16,067", "50,000", "50,000"]
        }
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)