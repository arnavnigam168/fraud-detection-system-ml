from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalMetrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float


def compute_metrics(y_true: pd.Series, y_proba: pd.Series, *, threshold: float) -> EvalMetrics:
    y_true = pd.Series(y_true).astype(int)
    y_proba = pd.Series(y_proba).astype(float)
    y_pred = (y_proba >= float(threshold)).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan")
    return EvalMetrics(precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)


def plot_confusion_matrix(y_true: pd.Series, y_proba: pd.Series, *, threshold: float):
    y_pred = (pd.Series(y_proba).astype(float) >= float(threshold)).astype(int)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc(y_true: pd.Series, y_proba: pd.Series):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    return fig


def threshold_impact(y_proba: pd.Series, *, steps: int = 25) -> pd.DataFrame:
    """Return a small table showing how many rows are flagged vs threshold."""
    p = pd.Series(y_proba).astype(float)
    thresholds = np.linspace(0.05, 0.95, steps)
    flagged = [(p >= t).sum() for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "flagged_transactions": flagged})

