"""Feature preparation for inference."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.features import (
    build_h1_trend_features,
    build_m15_features,
    get_feature_columns,
    merge_m15_with_h1,
)


def build_live_feature_row(
    df_m15: pd.DataFrame, df_h1: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Build the latest feature row from live M15/H1 candles.
    """
    if df_m15.empty or df_h1.empty:
        raise ValueError("Empty candle data for live features")

    m15_feats = build_m15_features(df_m15)
    h1_feats = build_h1_trend_features(df_h1)
    merged = merge_m15_with_h1(m15_feats, h1_feats)
    merged = merged.sort_values("time").reset_index(drop=True)

    last_row = merged.iloc[-1]
    cols = get_feature_columns()
    missing = [c for c in cols if c not in last_row.index]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    feature_df = pd.DataFrame([last_row[cols].values], columns=cols)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)
    if feature_df.isna().any().any():
        raise ValueError("NaNs in live feature row after fill")

    meta = {
        "time": last_row.get("time"),
        "close": last_row.get("close"),
        "adx_14": float(last_row.get("adx_14", 0.0)),
        "rsi_14": float(last_row.get("rsi_14", 0.0)),
        "ema_diff": float(last_row.get("ema_20_50_diff", 0.0)),
        "h1_flag": int(last_row.get("h1_trend_flag", 0)),
    }
    return feature_df, last_row, meta


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
