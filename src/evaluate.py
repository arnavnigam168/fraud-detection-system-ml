import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, confusion_matrix, roc_auc_score

from .preprocessing import clean_data, load_data, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_trained(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    payload = joblib.load(path)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return model


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_proba = (scores.max() - scores) / (scores.max() - scores.min() + 1e-8)
    y_pred = (y_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {"roc_auc": auc, "f1": report["1"]["f1-score"], "precision": report["1"]["precision"], "recall": report["1"]["recall"]}

    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    return metrics


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, out_path: Path) -> None:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_proba = (scores.max() - scores) / (scores.max() - scores.min() + 1e-8)
    y_pred = (y_proba >= 0.5).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_roc(model, X_test: pd.DataFrame, y_test: pd.Series, out_path: Path) -> None:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_proba = (scores.max() - scores) / (scores.max() - scores.min() + 1e-8)
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fraud model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with is_fraud target.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved joblib model.")
    parser.add_argument("--output_dir", type=str, default="reports", help="Where to save plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.data_path)
    df = clean_data(df, target_col="is_fraud")
    X_train, X_test, y_train, y_test = split_features_target(df, target_col="is_fraud", test_size=0.2)
    model = load_trained(args.model_path)

    metrics = evaluate(model, X_test, y_test)
    logger.info("Evaluation metrics: %s", metrics)

    output_dir = Path(args.output_dir)
    plot_confusion_matrix(model, X_test, y_test, output_dir / "confusion_matrix.png")
    plot_roc(model, X_test, y_test, output_dir / "roc_curve.png")
    logger.info("Saved plots to %s", output_dir)


if __name__ == "__main__":
    main()

