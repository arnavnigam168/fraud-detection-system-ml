import logging
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Fraud Detection", layout="wide")


@st.cache_resource
def load_model(model_path: str = "models/best_model.joblib"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError("Trained model not found. Please run training first.")
    payload = joblib.load(path)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    best_name = payload.get("best_model") if isinstance(payload, dict) else "model"
    return model, metrics, best_name


def predict(model, df: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[:, 1]
    else:
        scores = model.decision_function(df)
        probs = (scores.max() - scores) / (scores.max() - scores.min() + 1e-8)
    return pd.Series(probs, index=df.index, name="fraud_probability")


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_roc(y_true: pd.Series, y_proba: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    plt.title("ROC Curve")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def highlight_risk(df: pd.DataFrame, prob_col: str = "fraud_probability", threshold: float = 0.5):
    def color_row(row):
        color = "#ffb3b3" if row[prob_col] >= threshold else ""
        return [f"background-color: {color}"] * len(row)

    return df.style.apply(color_row, axis=1)


def main() -> None:
    st.write("Upload transaction data to score fraud risk. If `is_fraud` is present, evaluation plots will be shown.")

    try:
        model, metrics, best_name = load_model()
        st.success(f"Loaded model: {best_name}")
        if metrics:
            st.json(metrics)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    threshold = st.slider("Fraud threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    if uploaded:
        df = pd.read_csv(uploaded)
        if df.empty:
            st.warning("Uploaded file is empty.")
            return

        target_present = "is_fraud" in df.columns
        if target_present:
            y_true = df["is_fraud"]
            X = df.drop(columns=["is_fraud"])
        else:
            y_true = None
            X = df

        probs = predict(model, X)
        preds = (probs >= threshold).astype(int)

        results = df.copy()
        results["fraud_probability"] = probs
        results["fraud_prediction"] = preds

        st.subheader("Fraud Probabilities")
        st.dataframe(highlight_risk(results.sort_values(by="fraud_probability", ascending=False), threshold=threshold))

        st.subheader("Top High-Risk Transactions")
        st.dataframe(results.nlargest(10, "fraud_probability")[["fraud_probability"] + [c for c in df.columns if c != "fraud_probability"]])

        if target_present:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(y_true, preds)

            st.subheader("ROC Curve")
            plot_roc(y_true, probs)

            st.subheader("Metrics on Uploaded Data")
            st.write(
                {
                    "precision": float(precision_score(y_true, preds)),
                    "recall": float(recall_score(y_true, preds)),
                }
            )
        else:
            st.info("No `is_fraud` column provided. Showing scores only.")

    else:
        st.info("Awaiting CSV upload.")


if __name__ == "__main__":
    main()