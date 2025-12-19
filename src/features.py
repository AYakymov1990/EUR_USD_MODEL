"""Feature engineering utilities for M15 and H1 data."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


def build_m15_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build M15-level technical features for EUR/USD."""
    data = df.copy()

    ema_20 = EMAIndicator(close=data["close"], window=20).ema_indicator()
    ema_50 = EMAIndicator(close=data["close"], window=50).ema_indicator()

    data["ema_20"] = ema_20
    data["ema_50"] = ema_50
    data["ema_20_50_diff"] = ema_20 - ema_50
    data["rsi_14"] = RSIIndicator(close=data["close"], window=14).rsi()
    data["adx_14"] = ADXIndicator(
        high=data["high"], low=data["low"], close=data["close"], window=14
    ).adx()
    data["atr_14"] = AverageTrueRange(
        high=data["high"], low=data["low"], close=data["close"], window=14
    ).average_true_range()
    data["ret_1"] = data["close"].pct_change(1)
    data["ret_3"] = data["close"].pct_change(3)

    return data


def build_h1_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build H1-level trend features."""
    data = df.copy()
    data["ema_50_h1"] = EMAIndicator(close=data["close"], window=50).ema_indicator()
    data["h1_trend_flag"] = (data["close"] > data["ema_50_h1"]).astype(int)

    return data[["time", "ema_50_h1", "h1_trend_flag"]]


def merge_m15_with_h1(
    df_m15: pd.DataFrame,
    df_h1_trend: pd.DataFrame,
) -> pd.DataFrame:
    """Merge M15 features with H1 trend features using time-based alignment."""
    left = df_m15.sort_values("time")
    right = df_h1_trend.sort_values("time")
    merged = pd.merge_asof(left, right, on="time", direction="backward")
    return merged


def add_target(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Add forward-return target column for a given horizon (in M15 bars)."""
    data = df.copy()
    data["target"] = data["close"].shift(-horizon) / data["close"] - 1
    return data


def drop_na_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaNs coming from indicator warm-up and target shift."""
    return df.dropna().reset_index(drop=True)
