"""Simple backtesting utilities for model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


BARS_PER_YEAR = 252 * 24 * 4


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _extract_trades(df: pd.DataFrame, position_col: str) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    position = df[position_col].fillna(0)
    changes = position.diff().fillna(position)
    entry_idx = position[(changes != 0) & (position != 0)].index
    exit_idx = position[(changes != 0) & (position == 0)].index

    open_trade = None
    for idx in position.index:
        if idx in entry_idx:
            if open_trade is not None:
                open_trade["exit_time"] = df.loc[idx, "time"]
                open_trade["exit_price"] = df.loc[idx, "close"]
                open_trade["pnl"] = (
                    (open_trade["exit_price"] - open_trade["entry_price"])
                    / open_trade["entry_price"]
                ) * open_trade["direction"]
                trades.append(open_trade)
            open_trade = {
                "entry_time": df.loc[idx, "time"],
                "entry_price": df.loc[idx, "close"],
                "direction": int(position.loc[idx]),
                "exit_time": None,
                "exit_price": None,
                "pnl": None,
            }
        if idx in exit_idx and open_trade is not None:
            open_trade["exit_time"] = df.loc[idx, "time"]
            open_trade["exit_price"] = df.loc[idx, "close"]
            open_trade["pnl"] = (
                (open_trade["exit_price"] - open_trade["entry_price"])
                / open_trade["entry_price"]
            ) * open_trade["direction"]
            trades.append(open_trade)
            open_trade = None

    return pd.DataFrame(trades)


def backtest_long_short(
    df: pd.DataFrame,
    pred_col: str = "pred",
    price_col: str = "close",
    threshold: float = 0.0,
    cost_bps: float = 0.5,
) -> BacktestResult:
    """Run a simple long/short backtest with next-bar execution and costs."""
    data = df.sort_values("time").reset_index(drop=True).copy()
    data["actual_return"] = data[price_col].pct_change().fillna(0.0)

    signal = np.where(data[pred_col] > threshold, 1, 0)
    signal = np.where(data[pred_col] < -threshold, -1, signal)
    data["signal"] = signal
    data["position"] = data["signal"].shift(1).fillna(0)

    turnover = data["position"].diff().abs().fillna(data["position"].abs())
    cost = turnover * (cost_bps / 10000.0)
    data["strategy_return"] = data["position"] * data["actual_return"] - cost
    data["equity_curve"] = (1 + data["strategy_return"]).cumprod()

    total_return = data["equity_curve"].iloc[-1] - 1.0
    annualized_return = (1 + total_return) ** (BARS_PER_YEAR / len(data)) - 1.0
    annualized_vol = data["strategy_return"].std() * np.sqrt(BARS_PER_YEAR)
    sharpe = (
        float(annualized_return / annualized_vol) if annualized_vol and annualized_vol > 0 else 0.0
    )
    max_dd = _max_drawdown(data["equity_curve"])

    active = data["position"] != 0
    hit_rate = (
        (data.loc[active, "strategy_return"] > 0).mean() if active.any() else 0.0
    )

    trades = _extract_trades(data, "position")

    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_vol": float(annualized_vol) if not np.isnan(annualized_vol) else 0.0,
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "hit_rate": float(hit_rate),
        "trade_count": float(len(trades)),
    }

    return BacktestResult(
        equity_curve=data["equity_curve"],
        returns=data["strategy_return"],
        trades=trades,
        metrics=metrics,
    )
