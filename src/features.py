"""Feature engineering utilities for M15 and H1 data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


M15_EMA_FAST = 20
M15_EMA_SLOW = 50
M15_RSI_PERIOD = 14
M15_ADX_PERIOD = 14
M15_ATR_PERIOD = 14
M15_RET_VOL_WINDOW = 20
ADX_TREND_THRESHOLD = 25.0
H1_EMA_PERIOD = 50
LOG_RET_NORM_CLIP = 10.0
VOL_Z_WINDOW = 50
DOW_PERIOD = 7


def build_m15_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build M15-level technical features for EUR/USD.

    Adds indicators, returns, volatility, time-of-day, and price-action features,
    including: range_rel, body_rel, upper_wick_rel, lower_wick_rel, volume_z,
    sin_dow, cos_dow.
    """
    data = df.copy()

    ema_20 = EMAIndicator(close=data["close"], window=M15_EMA_FAST).ema_indicator()
    ema_50 = EMAIndicator(close=data["close"], window=M15_EMA_SLOW).ema_indicator()

    data["ema_20"] = ema_20
    data["ema_50"] = ema_50
    data["ema_20_50_diff"] = ema_20 - ema_50
    data["rsi_14"] = RSIIndicator(close=data["close"], window=M15_RSI_PERIOD).rsi()
    data["adx_14"] = ADXIndicator(
        high=data["high"], low=data["low"], close=data["close"], window=M15_ADX_PERIOD
    ).adx()
    data["atr_14"] = AverageTrueRange(
        high=data["high"], low=data["low"], close=data["close"], window=M15_ATR_PERIOD
    ).average_true_range()
    data["ret_1"] = data["close"].pct_change(1)
    data["ret_3"] = data["close"].pct_change(3)

    data["log_ret_1"] = np.log(data["close"] / data["close"].shift(1))
    data["log_ret_3"] = np.log(data["close"] / data["close"].shift(3))
    data["roll_vol_20"] = data["log_ret_1"].rolling(M15_RET_VOL_WINDOW).std()
    data["log_ret_1_norm"] = data["log_ret_1"] / data["roll_vol_20"].replace(0, np.nan)
    data["log_ret_1_norm"] = data["log_ret_1_norm"].clip(
        -LOG_RET_NORM_CLIP, LOG_RET_NORM_CLIP
    )
    data["atr_14_norm"] = data["atr_14"] / data["close"]
    data["trend_strength_m15"] = data["ema_20_50_diff"] / data["atr_14"].replace(0, np.nan)

    close_safe = data["close"].replace(0, np.nan)
    data["range_rel"] = (data["high"] - data["low"]) / close_safe
    data["body_rel"] = (data["close"] - data["open"]) / close_safe
    data["upper_wick_rel"] = (
        data["high"] - data[["open", "close"]].max(axis=1)
    ) / close_safe
    data["lower_wick_rel"] = (
        data[["open", "close"]].min(axis=1) - data["low"]
    ) / close_safe
    data["range_rel"] = data["range_rel"].clip(lower=0)
    data["upper_wick_rel"] = data["upper_wick_rel"].clip(lower=0)
    data["lower_wick_rel"] = data["lower_wick_rel"].clip(lower=0)

    volume_mean = data["volume"].rolling(VOL_Z_WINDOW).mean()
    volume_std = data["volume"].rolling(VOL_Z_WINDOW).std()
    data["volume_z"] = (data["volume"] - volume_mean) / volume_std.replace(0, np.nan)

    data["hour"] = data["time"].dt.hour
    data["minute"] = data["time"].dt.minute
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow"] = data["time"].dt.dayofweek
    data["sin_dow"] = np.sin(2 * np.pi * data["dow"] / DOW_PERIOD)
    data["cos_dow"] = np.cos(2 * np.pi * data["dow"] / DOW_PERIOD)
    data["adx_above_threshold"] = (data["adx_14"] > ADX_TREND_THRESHOLD).astype(int)

    return data


def build_h1_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build H1-level trend features."""
    data = df.copy()
    data["ema_50_h1"] = EMAIndicator(close=data["close"], window=H1_EMA_PERIOD).ema_indicator()
    data["h1_trend_flag"] = (data["close"] > data["ema_50_h1"]).astype(int)
    data["h1_trend_distance"] = data["close"] / data["ema_50_h1"] - 1

    return data[["time", "ema_50_h1", "h1_trend_flag", "h1_trend_distance"]]


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


def validate_no_nans(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise a ValueError if any NaNs exist in the specified columns."""
    na_counts = df[cols].isna().sum()
    bad = na_counts[na_counts > 0]
    if not bad.empty:
        raise ValueError(f"NaNs detected in columns: {bad.to_dict()}")


def price_action_sanity_checks(df: pd.DataFrame) -> None:
    """Basic sanity checks for price-action features."""
    cols = ["range_rel", "upper_wick_rel", "lower_wick_rel"]
    if (df[cols] < -1e-12).any().any():
        raise ValueError("Negative wick/range values detected beyond tolerance.")


def get_feature_columns() -> list[str]:
    """Return the list of feature column names to be used for model training."""
    return [
        "ema_20",
        "ema_50",
        "ema_20_50_diff",
        "rsi_14",
        "adx_14",
        "atr_14",
        "atr_14_norm",
        "ret_1",
        "ret_3",
        "log_ret_1",
        "log_ret_3",
        "roll_vol_20",
        "log_ret_1_norm",
        "trend_strength_m15",
        "range_rel",
        "body_rel",
        "upper_wick_rel",
        "lower_wick_rel",
        "volume_z",
        "ema_50_h1",
        "h1_trend_flag",
        "h1_trend_distance",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "adx_above_threshold",
    ]
