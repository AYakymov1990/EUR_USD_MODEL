"""Simple backtesting logic based on model predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_strategy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Run a basic long/short backtest based on predictions.

    Args:
        df: DataFrame with 'close' prices.
        predictions: Model predictions aligned with df rows.
        threshold: Decision threshold for entering positions.

    Returns:
        DataFrame with signals and strategy returns.
    """
    bt = df.copy()
    bt["prediction"] = predictions
    bt["signal"] = 0
    bt.loc[bt["prediction"] > threshold, "signal"] = 1
    bt.loc[bt["prediction"] < -threshold, "signal"] = -1

    bt["return_1"] = bt["close"].pct_change()
    bt["strategy_return"] = bt["signal"].shift(1) * bt["return_1"]
    bt["equity_curve"] = (1 + bt["strategy_return"].fillna(0)).cumprod()

    return bt
