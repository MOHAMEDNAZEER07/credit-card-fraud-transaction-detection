"""
Credit Card Fraud Detection System
-----------------------------------
A Streamlit app that trains a logistic regression model on the Kaggle
Credit Card Fraud dataset and lets users upload transaction data to
flag likely fraudulent transactions.
"""

import logging
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# Config / constants
# -------------------------------------------------------------------
DATA_PATH = "credit/creditcard.csv"
MODEL_CACHE_PATH = "credit/model.joblib"
SCALER_CACHE_PATH = "credit/scaler.joblib"
RANDOM_STATE = 2
MAX_UPLOAD_ROWS = 200_000  # safety cap to avoid freezing the UI
TARGET_COLUMN = "Class"
AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data loading & model training (cached)
# -------------------------------------------------------------------
@st.cache_resource
def prepare_and_train(
    use_class_weight: bool = False,
) -> Tuple[LogisticRegression, StandardScaler, List[str], dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset, balance classes (or use class_weight), scale features,
    train a logistic regression model, and evaluate it.

    Returns:
        model, scaler, expected_columns, metrics_dict, full_data, legit_df, fraud_df
    """
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logger.error("Dataset not found at %s", DATA_PATH)
        st.error(f"Could not find dataset at '{DATA_PATH}'. Please check the file path.")
        st.stop()

    legit = data[data[TARGET_COLUMN] == 0]
    fraud = data[data[TARGET_COLUMN] == 1]

    if use_class_weight:
        # Use the full dataset, let the model handle imbalance internally
        train_df = data
    else:
        # Undersample legitimate transactions to balance classes
        legit_sample = legit.sample(n=len(fraud), random_state=RANDOM_STATE)
        train_df = pd.concat([legit_sample, fraud], axis=0)

    X = train_df.drop(columns=TARGET_COLUMN, axis=1)
    y = train_df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced" if use_class_weight else None,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "train_acc": accuracy_score(y_train, model.predict(X_train)),
        "test_acc": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    # Persist model + scaler to disk so future cold starts can skip retraining
    try:
        os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
        joblib.dump(model, MODEL_CACHE_PATH)
        joblib.dump(scaler, SCALER_CACHE_PATH)
        logger.info("Model and scaler persisted to disk.")
    except Exception as e:
        logger.warning("Could not persist model to disk: %s", e)

    return model, scaler, X.columns.tolist(), metrics, data, legit, fraud


def validate_uploaded_data(df: pd.DataFrame, expected_columns: List[str]) -> Tuple[bool, str]:
    """Validate an uploaded CSV before running predictions."""
    if df.empty:
        return False, "The uploaded file is empty."
    if len(df) > MAX_UPLOAD_ROWS:
        return False, f"File has {len(df):,} rows, which exceeds the {MAX_UPLOAD_ROWS:,} row limit."
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    if df[expected_columns].isnull().any().any():
        return False, "The uploaded file contains missing values in required columns. Please clean the data first."
    return True, ""


def generate_sample_template(columns: List[str]) -> str:
    """Generate a one-row sample CSV template matching the expected schema."""
    sample = pd.DataFrame([[0] * len(columns)], columns=columns)
    return sample.to_csv(index=False)


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    """
    Welcome to the **Credit Card Fraud Detection System**. This app uses a machine learning
    model to predict fraudulent transactions based on historical data.
    """
)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    use_class_weight = st.checkbox(
        "Use full dataset with class balancing (instead of undersampling)",
        value=False,
        help="When off, the model trains on an undersampled, balanced subset. "
             "When on, it trains on all data and uses class_weight='balanced'.",
    )

    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Transactions with predicted fraud probability above this value are flagged as fraudulent.",
    )

    st.divider()
    st.header("📁 Upload")
    uploaded_file = st.file_uploader("Upload a CSV file with transaction features", type="csv")

# ---------------- Train / load model ----------------
with st.spinner("Loading data and training model..."):
    model, scaler, expected_columns, metrics, data, legit, fraud = prepare_and_train(use_class_weight)

# ---------------- Model performance ----------------
st.subheader("📈 Model Performance")
with st.container():
    cols = st.columns(5)
    cols[0].metric("Train Accuracy", f"{metrics['train_acc']:.2%}")
    cols[1].metric("Test Accuracy", f"{metrics['test_acc']:.2%}")
    cols[2].metric("Precision", f"{metrics['precision']:.2%}")
    cols[3].metric("Recall", f"{metrics['recall']:.2%}")
    cols[4].metric("F1 Score", f"{metrics['f1']:.2%}")

with st.expander("🔎 Confusion Matrix (test set)"):
    cm = metrics["confusion_matrix"]
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
        xticklabels=["Legitimate", "Fraudulent"],
        yticklabels=["Legitimate", "Fraudulent"],
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    st.caption(
        "Accuracy alone can be misleading on imbalanced fraud data — "
        "precision/recall/F1 and the confusion matrix give a fuller picture."
    )

# ---------------- Sample template download ----------------
st.subheader("📄 Need a template?")
sample_csv = generate_sample_template(expected_columns + [TARGET_COLUMN])
st.download_button(
    label="Download sample CSV template",
    data=sample_csv,
    file_name="sample_transactions_template.csv",
    mime="text/csv",
)

# ---------------- Upload & prediction ----------------
st.subheader("🔍 Upload Your Data")

if uploaded_file is not None:
    try:
        with st.spinner("Reading uploaded file..."):
            input_data = pd.read_csv(uploaded_file)
            input_data = input_data.drop(columns=[TARGET_COLUMN], errors="ignore")

        is_valid, error_msg = validate_uploaded_data(input_data, expected_columns)

        if not is_valid:
            st.error(error_msg)
        else:
            progress_bar = st.progress(0, text="Preparing data...")

            input_data = input_data[expected_columns]  # reorder to match training schema
            progress_bar.progress(30, text="Scaling features...")

            input_data_scaled = scaler.transform(input_data)
            progress_bar.progress(60, text="Running predictions...")

            fraud_probabilities = model.predict_proba(input_data_scaled)[:, 1]
            predictions = (fraud_probabilities >= threshold).astype(int)
            progress_bar.progress(90, text="Compiling results...")

            input_data["Fraud Probability"] = fraud_probabilities
            input_data["Prediction"] = predictions
            input_data["Result"] = input_data["Prediction"].apply(
                lambda x: "Legitimate" if x == 0 else "Fraudulent"
            )

            progress_bar.progress(100, text="Done!")
            progress_bar.empty()

            # ---------------- Fraud analysis ----------------
            fraudulent_transactions = input_data[input_data["Result"] == "Fraudulent"]
            total_transactions = len(input_data)
            fraudulent_transactions_count = len(fraudulent_transactions)
            fraud_percentage = (
                (fraudulent_transactions_count / total_transactions) * 100
                if total_transactions > 0 else 0
            )
            total_fraud_amount = fraudulent_transactions[AMOUNT_COLUMN].sum()

            display_columns = [TIME_COLUMN, AMOUNT_COLUMN, "Fraud Probability", "Result"]
            display_columns = [c for c in display_columns if c in fraudulent_transactions.columns]
            fraudulent_transactions_display = fraudulent_transactions[display_columns]

            with st.expander("📊 Fraud Analysis", expanded=True):
                cols = st.columns(4)
                cols[0].metric("Total Transactions", f"{total_transactions:,}")
                cols[1].metric("Fraudulent Transactions", f"{fraudulent_transactions_count:,}")
                cols[2].metric("Fraud Rate", f"{fraud_percentage:.2f}%")
                cols[3].metric("Total Fraud Amount", f"${total_fraud_amount:,.2f}")

                st.subheader("Fraudulent Transactions Data")
                st.dataframe(fraudulent_transactions_display, use_container_width=True)

                csv_download = fraudulent_transactions_display.to_csv(index=False)
                st.download_button(
                    label="Download Fraudulent Transactions",
                    data=csv_download,
                    file_name="fraudulent_transactions.csv",
                    mime="text/csv",
                )

            # ---------------- Visualizations ----------------
            with st.expander("📈 Visualizations"):
                col_a, col_b = st.columns(2)

                with col_a:
                    # Pie Chart
                    fig1, ax1 = plt.subplots()
                    ax1.pie(
                        [len(legit), len(fraud)],
                        labels=["Legitimate", "Fraudulent"],
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=["#66b3ff", "#ff9999"],
                    )
                    ax1.axis("equal")
                    ax1.set_title("Training Data Class Balance")
                    st.pyplot(fig1)

                with col_b:
                    # Bar Chart
                    fig3, ax3 = plt.subplots()
                    counts = data[TARGET_COLUMN].value_counts().sort_index()
                    counts.plot(kind="bar", ax=ax3, color=["#66b3ff", "#ff9999"])
                    ax3.set_title("Fraudulent vs Legitimate (Full Dataset)")
                    ax3.set_xticks(range(len(counts)))
                    ax3.set_xticklabels(["Legitimate", "Fraudulent"], rotation=0)
                    ax3.set_ylabel("Count")
                    st.pyplot(fig3)

                # Boxplot
                fig2, ax2 = plt.subplots()
                sns.boxplot(
                    x=TARGET_COLUMN, y=AMOUNT_COLUMN, data=data, ax=ax2,
                    hue=TARGET_COLUMN, palette="Set2", legend=False,
                )
                ax2.set_title("Transaction Amount Distribution")
                ax2.set_xticks([0, 1])
                ax2.set_xticklabels(["Legitimate", "Fraudulent"])
                st.pyplot(fig2)

                # Correlation Heatmap
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                corr_matrix = data.corr(numeric_only=True)
                sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax4)
                ax4.set_title("Feature Correlation Heatmap")
                st.pyplot(fig4)

                # Histogram
                fig5, ax5 = plt.subplots()
                sns.histplot(
                    data=data, x=AMOUNT_COLUMN, hue=TARGET_COLUMN, kde=True, ax=ax5,
                    palette={0: "green", 1: "red"},
                )
                ax5.set_title("Transaction Amounts Distribution")
                ax5.set_xlabel("Amount")
                st.pyplot(fig5)

                # Line Chart: Fraudulent Transactions Over Time
                if TIME_COLUMN in input_data.columns:
                    fraud_over_time = fraudulent_transactions.groupby(TIME_COLUMN).size()
                    fig6, ax6 = plt.subplots()
                    fraud_over_time.plot(kind="line", ax=ax6, color="red")
                    ax6.set_title("Fraudulent Transactions Over Time")
                    ax6.set_xlabel("Time")
                    ax6.set_ylabel("Fraudulent Transactions")
                    st.pyplot(fig6)

    except Exception as e:
        logger.exception("Error while processing uploaded file")
        st.error(f"Something went wrong while processing your file: {e}")

else:
    st.info("Please upload a CSV file to see predictions.")
