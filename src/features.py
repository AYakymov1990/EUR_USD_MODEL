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


def build_m15_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build M15-level technical features for EUR/USD."""
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

    data["hour"] = data["time"].dt.hour
    data["minute"] = data["time"].dt.minute
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)
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


def add_target(
    df: pd.DataFrame,
    horizon: int = 3,
    price_col: str = "close",
    target_col: str = "target",
) -> pd.DataFrame:
    """Add forward-return target column for a given horizon (in M15 bars).

    Execution enters on the next bar and exits after `horizon` bars,
    matching backtest_long_short_horizon semantics.
    """
    data = df.copy()
    base = data[price_col].shift(-1)
    fut = data[price_col].shift(-(1 + horizon))
    data[target_col] = fut / base - 1
    return data


def check_target_alignment(
    df: pd.DataFrame,
    horizon: int,
    price_col: str = "close",
    target_col: str = "target",
    eps: float = 1e-12,
) -> float:
    """Print and assert alignment between target and execution returns."""
    exec_ret = df[price_col].shift(-(1 + horizon)) / df[price_col].shift(-1) - 1
    mask = df[target_col].notna() & exec_ret.notna()
    diff = (df.loc[mask, target_col] - exec_ret.loc[mask]).abs()
    max_abs_diff = float(diff.max()) if len(diff) else 0.0
    print(f"max_abs_diff: {max_abs_diff}")
    if max_abs_diff > eps:
        raise AssertionError(f"Target misalignment: max_abs_diff={max_abs_diff}")
    return max_abs_diff


def drop_na_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaNs coming from indicator warm-up and target shift."""
    return df.dropna().reset_index(drop=True)


def validate_no_nans(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise a ValueError if any NaNs exist in the specified columns."""
    na_counts = df[cols].isna().sum()
    bad = na_counts[na_counts > 0]
    if not bad.empty:
        raise ValueError(f"NaNs detected in columns: {bad.to_dict()}")


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
        "ema_50_h1",
        "h1_trend_flag",
        "h1_trend_distance",
        "sin_hour",
        "cos_hour",
        "adx_above_threshold",
    ]
