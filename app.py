from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from evaluation import compute_metrics, plot_confusion_matrix, plot_roc, threshold_impact
from model import (
    ModelPayload,
    available_model_paths,
    expected_raw_feature_columns,
    load_model_payload,
    predict_fraud_probability,
    try_random_forest_feature_importance,
)
from preprocessing import validate_and_prepare

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_data
def _read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _metric_help() -> str:
    return (
        "Precision: among flagged as fraud, how many are truly fraud.\n\n"
        "Recall: among true fraud, how many we catch.\n\n"
        "F1: balance of precision and recall.\n\n"
        "ROC-AUC: ranking quality across thresholds (1.0 is best)."
    )


def _render_header() -> None:
    st.title("🛡️ Fraud Detection Dashboard")
    st.caption(
        "Upload transactions, score fraud risk, and (optionally) evaluate performance when ground-truth `is_fraud` is present. "
        "Designed for clean academic demonstration: clear flow, robust validation, and explainable outputs."
    )
    st.divider()


def _render_sidebar(model_paths: dict[str, "Path"]) -> dict:
    st.sidebar.header("⚙️ Controls")

    upload = st.sidebar.file_uploader("📤 Upload transactions CSV", type=["csv"])

    model_key = st.sidebar.selectbox(
        "🧠 Model selection",
        options=(["best_model"] + [k for k in ["log_reg", "random_forest", "isolation_forest"] if k in model_paths]),
        format_func=lambda k: {
            "best_model": "Best saved model (recommended)",
            "log_reg": "Logistic Regression",
            "random_forest": "Random Forest",
            "isolation_forest": "Isolation Forest (anomaly)",
        }.get(k, k),
    )

    threshold = st.sidebar.slider("🎚️ Fraud threshold", 0.05, 0.95, 0.50, 0.01)
    top_n = st.sidebar.slider("🔎 Show top risky transactions", 5, 50, 10, 5)

    st.sidebar.divider()
    show_explain = st.sidebar.checkbox("📚 Show faculty explanation", value=True)
    show_threshold_impact = st.sidebar.checkbox("📈 Show threshold impact", value=True)
    allow_download = st.sidebar.checkbox("⬇️ Enable results download", value=True)

    return {
        "upload": upload,
        "model_key": model_key,
        "threshold": float(threshold),
        "top_n": int(top_n),
        "show_explain": bool(show_explain),
        "show_threshold_impact": bool(show_threshold_impact),
        "allow_download": bool(allow_download),
    }


def _render_model_comparison(payload: ModelPayload) -> None:
    st.subheader("📊 Model comparison (from training run)")
    if not payload.metrics:
        st.info("No training metrics were packaged with the saved model artifact.")
        return
    df = (
        pd.DataFrame(payload.metrics)
        .T.rename_axis("model")
        .reset_index()
        .sort_values(["roc_auc", "f1"], ascending=False, na_position="last")
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_explainability(payload: ModelPayload, *, threshold: float) -> None:
    st.subheader("📚 Explainability (for viva / demo)")

    st.markdown(
        f"""
**Which model is used?** `{payload.model_name}` (loaded from `{payload.source_path.as_posix()}`)

**Why this model?**
- During training, we compare multiple algorithms and select the best-performing one using validation metrics (ROC-AUC and F1).
- The saved artifact contains the chosen pipeline including preprocessing, so inference uses the same transformations as training.

**How to interpret the score?**
- Each transaction gets a fraud risk score in \\([0, 1]\\).
- A transaction is *flagged* as fraud when `fraud_probability ≥ {threshold:.2f}`.
        """.strip()
    )

    imp = try_random_forest_feature_importance(payload.model)
    if imp is not None and not imp.empty:
        with st.expander("🌲 Optional: Random Forest feature importance", expanded=False):
            st.caption("Higher importance means the feature contributed more to the model’s splits (after preprocessing).")
            st.dataframe(imp.head(20), use_container_width=True, hide_index=True)


def main() -> None:
    _render_header()

    model_paths = available_model_paths("models")
    controls = _render_sidebar(model_paths)

    # Load selected model artifact.
    try:
        if controls["model_key"] == "best_model":
            payload = load_model_payload("models/best_model.joblib")
        else:
            # Optional artifacts if you export them.
            payload = load_model_payload(str(model_paths[controls["model_key"]]))
    except Exception as e:
        st.error(str(e))
        st.info("If you haven't trained yet, run `python -m src.model --data_path data/transactions.csv --model_dir models`.")
        return

    left, right = st.columns([2, 1], vertical_alignment="top")
    with left:
        st.success(f"✅ Model loaded: **{payload.model_name}**")
    with right:
        st.caption("🧾 Tip: include `is_fraud` in your CSV to enable evaluation plots & metrics.")

    st.divider()

    if payload.metrics:
        _render_model_comparison(payload)
        st.divider()

    uploaded = controls["upload"]
    if uploaded is None:
        st.info("📤 Upload a CSV from the sidebar to begin.")
        return

    try:
        df = _read_csv_bytes(uploaded.getvalue())
    except Exception:
        st.error("Could not read the uploaded file as a CSV. Please upload a valid `.csv`.")
        return

    expected_cols = expected_raw_feature_columns(payload.model)

    try:
        X, y_true, report = validate_and_prepare(df, expected_columns=expected_cols, target_col="is_fraud")
    except Exception as e:
        st.error(str(e))
        if expected_cols:
            with st.expander("Expected feature schema", expanded=False):
                st.write(pd.DataFrame({"expected_columns": expected_cols}))
        return

    if report.dropped_extra:
        st.warning(
            "Dropped extra columns not used by the trained pipeline: " + ", ".join(report.dropped_extra[:25])
            + (" ..." if len(report.dropped_extra) > 25 else "")
        )

    # Predict
    try:
        proba = predict_fraud_probability(payload.model, X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    threshold = controls["threshold"]
    pred = (proba >= threshold).astype(int).rename("fraud_prediction")

    results = df.copy()
    results["fraud_probability"] = proba
    results["fraud_prediction"] = pred

    st.subheader("📌 Key metrics")
    if y_true is not None:
        m = compute_metrics(y_true, proba, threshold=threshold)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Precision", f"{m.precision:.3f}", help=_metric_help())
        c2.metric("🧲 Recall", f"{m.recall:.3f}", help=_metric_help())
        c3.metric("⚖️ F1-score", f"{m.f1:.3f}", help=_metric_help())
        c4.metric("📈 ROC-AUC", "N/A" if pd.isna(m.roc_auc) else f"{m.roc_auc:.3f}", help=_metric_help())
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎚️ Threshold", f"{threshold:.2f}")
        c2.metric("🚩 Flagged transactions", int((proba >= threshold).sum()))
        c3.metric("📄 Total transactions", int(len(results)))
        c4.metric("🧠 Model", payload.model_name)
        st.info("Metrics require ground-truth labels. Add an `is_fraud` column (0/1) to your CSV to evaluate.")

    st.divider()

    st.subheader("🚨 Top suspicious transactions")
    top_n = controls["top_n"]
    top = results.sort_values("fraud_probability", ascending=False).head(top_n)
    st.dataframe(top, use_container_width=True, hide_index=True)

    if controls["allow_download"]:
        st.download_button(
            "⬇️ Download full results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
            use_container_width=False,
        )

    st.divider()

    if y_true is not None:
        st.subheader("🧪 Evaluation visuals")
        p1, p2 = st.columns(2)
        with p1:
            fig_cm = plot_confusion_matrix(y_true, proba, threshold=threshold)
            st.pyplot(fig_cm, use_container_width=True)
        with p2:
            fig_roc = plot_roc(y_true, proba)
            st.pyplot(fig_roc, use_container_width=True)

    if controls["show_threshold_impact"]:
        st.subheader("📈 Threshold tuning impact")
        impact = threshold_impact(proba, steps=25)
        st.line_chart(impact.set_index("threshold"))
        st.caption("As you increase the threshold, fewer transactions are flagged (precision tends to rise, recall tends to fall).")

    st.divider()

    if controls["show_explain"]:
        _render_explainability(payload, threshold=threshold)


if __name__ == "__main__":
    main()