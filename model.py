from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class ModelPayload:
    model: Any
    model_name: str
    metrics: Dict[str, Dict[str, float]]
    source_path: Path


def _normalize_payload(payload: Any, source_path: Path) -> ModelPayload:
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        metrics = payload.get("metrics") or {}
        model_name = payload.get("best_model") or "best_model"
        return ModelPayload(model=model, model_name=str(model_name), metrics=dict(metrics), source_path=source_path)
    return ModelPayload(model=payload, model_name="model", metrics={}, source_path=source_path)


@st.cache_resource
def load_model_payload(model_path: str = "models/best_model.joblib") -> ModelPayload:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Train first (example: `python -m src.model --data_path ... --model_dir models`)."
        )
    payload = joblib.load(path)
    return _normalize_payload(payload, source_path=path)


def available_model_paths(models_dir: str | Path = "models") -> Dict[str, Path]:
    """
    Best-effort discovery for optional per-model artifacts.
    Training currently saves only `best_model.joblib`, but the app can support
    additional files if you export them later.
    """
    d = Path(models_dir)
    candidates = {
        "best_model": d / "best_model.joblib",
        "log_reg": d / "log_reg.joblib",
        "random_forest": d / "random_forest.joblib",
        "isolation_forest": d / "isolation_forest.joblib",
    }
    return {k: v for k, v in candidates.items() if v.exists()}


def expected_raw_feature_columns(model: Any) -> Optional[list[str]]:
    """
    Infer the *raw input* feature columns expected by the saved pipeline.
    Works for the repo's ImbPipeline with a ColumnTransformer named `preprocess`.
    Returns None if inference is not possible.
    """
    preprocess = getattr(model, "named_steps", {}).get("preprocess") if hasattr(model, "named_steps") else None
    if preprocess is None:
        return list(getattr(model, "feature_names_in_", [])) or None

    cols: list[str] = []
    transformers = getattr(preprocess, "transformers_", None)
    if not transformers:
        return None

    for _, _, col_spec in transformers:
        if isinstance(col_spec, (list, tuple, np.ndarray, pd.Index)):
            cols.extend([str(c) for c in col_spec])
        elif isinstance(col_spec, slice):
            # Rare in this project; cannot reliably map to names without training-time context.
            return None
        else:
            # "drop" / "passthrough" / callable, etc.
            continue

    return sorted(set(cols)) if cols else None


def predict_fraud_probability(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    Return a Series in [0, 1] interpreted as "fraud probability / risk score".
    - For classifiers: uses `predict_proba` when available.
    - For IsolationForest-like models: converts anomaly score to a risk score.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=X.index, name="fraud_probability")

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        # Higher anomaly -> higher fraud risk
        probs = (scores.max() - scores) / (scores.max() - scores.min() + 1e-12)
        return pd.Series(probs, index=X.index, name="fraud_probability")

    raise TypeError("Loaded model does not support probability or scoring prediction.")


def try_random_forest_feature_importance(model: Any) -> Optional[pd.DataFrame]:
    """
    If the model is a RandomForest inside the project's pipeline, return a tidy
    importance table. Otherwise return None.
    """
    if not hasattr(model, "named_steps"):
        return None
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return None

    preprocess = model.named_steps.get("preprocess")
    if preprocess is None or not hasattr(preprocess, "get_feature_names_out"):
        return None

    try:
        names = preprocess.get_feature_names_out()
        importances = clf.feature_importances_
        imp = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
        imp["importance"] = imp["importance"].astype(float)
        return imp.reset_index(drop=True)
    except Exception:
        return None

