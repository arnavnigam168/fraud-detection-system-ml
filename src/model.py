import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import clean_data, load_data, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "iufc"]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    if not transformers:
        raise ValueError("No features available to preprocess.")

    return ColumnTransformer(transformers)


def build_models(preprocessor: ColumnTransformer) -> Dict[str, ImbPipeline]:
    smote = SMOTE(random_state=42)

    log_reg = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", smote),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    rf = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", smote),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )

    iso_forest = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", IsolationForest(n_estimators=200, contamination="auto", random_state=42)),
        ]
    )

    return {
        "log_reg": log_reg,
        "random_forest": rf,
        "isolation_forest": iso_forest,
    }


def evaluate_model(
    name: str, model: ImbPipeline, X_val: pd.DataFrame, y_val: pd.Series
) -> Tuple[float, float]:
    if name == "isolation_forest":
        scores = model.decision_function(X_val)
        fraud_probs = (scores.max() - scores) / (scores.max() - scores.min() + 1e-8)
        y_pred = (fraud_probs >= 0.5).astype(int)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, fraud_probs)
        return f1, auc

    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = model.decision_function(X_val)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    return f1, auc


def train_and_select(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    model_dir: str | Path = "models",
) -> Dict[str, Dict[str, float]]:
    X_train, X_val, y_train, y_val = split_features_target(df, target_col=target_col, test_size=0.2)
    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    metrics: Dict[str, Dict[str, float]] = {}
    best_model_name = None
    best_auc = -np.inf
    best_model = None

    for name, model in models.items():
        logger.info("Training model: %s", name)
        model.fit(X_train, y_train)
        f1, auc = evaluate_model(name, model, X_val, y_val)
        metrics[name] = {"f1": f1, "roc_auc": auc}
        logger.info("Model %s -> F1=%.4f ROC-AUC=%.4f", name, f1, auc)

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "best_model.joblib"
    joblib.dump({"model": best_model, "metrics": metrics, "best_model": best_model_name}, save_path)
    logger.info("Saved best model '%s' to %s", best_model_name, save_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with is_fraud target.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.data_path)
    df = clean_data(df, target_col="is_fraud")
    metrics = train_and_select(df, target_col="is_fraud", model_dir=args.model_dir)
    logger.info("Training complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()

