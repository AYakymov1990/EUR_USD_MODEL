"""Feature preparation for inference."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.features import get_feature_columns


def build_feature_row(row: pd.Series) -> pd.DataFrame:
    """
    Prepare a single-row DataFrame with the required feature columns.

    Assumes demo rows already contain the engineered features.
    """
    cols = get_feature_columns()
    missing = [c for c in cols if c not in row.index]
    if missing:
        raise ValueError(f"Missing feature columns in row: {missing}")
    df = pd.DataFrame([row[cols].values], columns=cols)
    return df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)


def batch_from_iter(rows: Iterable[pd.Series]) -> pd.DataFrame:
    """Convert an iterable of Series to a feature DataFrame."""
    cols = get_feature_columns()
    data = []
    for row in rows:
        data.append([row.get(c, np.nan) for c in cols])
    return pd.DataFrame(data, columns=cols)
