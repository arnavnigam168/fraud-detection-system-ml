import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_data(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(path)
    logger.info("Loaded data shape: %s", df.shape)
    return df


def clean_data(df: pd.DataFrame, target_col: str = "is_fraud") -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    before = df.shape[0]
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - df.shape[0])

    y = df[target_col]
    X = df.drop(columns=[target_col])

    for col in X.columns:
        if X[col].dtype.kind in "iufc":
            median = X[col].median()
            X[col] = X[col].fillna(median)
        else:
            mode = X[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "missing"
            X[col] = X[col].fillna(fill_value)

    cleaned = pd.concat([X, y], axis=1)
    return cleaned


def split_features_target(
    df: pd.DataFrame, target_col: str = "is_fraud", test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(
        "Split data: train=%s, test=%s, positive rate train=%.4f test=%.4f",
        X_train.shape,
        X_test.shape,
        y_train.mean(),
        y_test.mean(),
    )
    return X_train, X_test, y_train, y_test

