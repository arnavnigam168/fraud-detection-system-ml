from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SchemaReport:
    expected_columns: Optional[list[str]]
    missing_required: list[str]
    dropped_extra: list[str]


def _fill_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            val = float(s.median()) if s.notna().any() else 0.0
            X[col] = s.fillna(val)
        else:
            mode = s.mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "missing"
            X[col] = s.fillna(fill).astype(object)
    return X


def validate_and_prepare(
    df: pd.DataFrame,
    *,
    expected_columns: Optional[Iterable[str]] = None,
    target_col: str = "is_fraud",
) -> Tuple[pd.DataFrame, Optional[pd.Series], SchemaReport]:
    """
    Validate the uploaded schema and return (X, y, report).

    - If `expected_columns` is provided, the returned X is restricted to those columns
      in the correct order, and extra columns are dropped.
    - If `target_col` exists in df, y is returned; otherwise y is None.
    - Missing values are imputed (median for numeric, mode/"missing" for categorical).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Uploaded file is empty.")

    y: Optional[pd.Series] = df[target_col] if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors="ignore")

    expected = list(expected_columns) if expected_columns is not None else None
    missing_required: list[str] = []
    dropped_extra: list[str] = []

    if expected is not None and len(expected) > 0:
        expected_set = set(expected)
        present_set = set(X.columns.astype(str))

        missing_required = sorted([c for c in expected if c not in present_set])
        if missing_required:
            raise ValueError(
                "Uploaded CSV is missing required feature columns: "
                + ", ".join(missing_required)
                + ". Please upload data with the same schema used during training."
            )

        dropped_extra = sorted([c for c in X.columns if str(c) not in expected_set])
        X = X[[c for c in expected if c in X.columns]]

    # Basic cleanup: duplicate rows and missing handling.
    X = X.drop_duplicates()
    if y is not None:
        y = y.loc[X.index]

    X = _fill_missing_values(X)

    report = SchemaReport(
        expected_columns=expected,
        missing_required=missing_required,
        dropped_extra=dropped_extra,
    )
    return X, y, report

