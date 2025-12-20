"""Backtesting utilities for horizon-aware model predictions."""

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


def _extract_trades_horizon(
    df: pd.DataFrame,
    position_col: str,
    horizon: int,
) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    position = df[position_col].fillna(0)

    entry_idx = position[(position != 0) & (position.shift(1).fillna(0) == 0)].index
    for idx in entry_idx:
        exit_idx = idx + horizon
        if exit_idx >= len(df):
            continue
        direction = int(position.loc[idx])
        entry_price = df.loc[idx, "close"]
        exit_price = df.loc[exit_idx, "close"]
        pnl = (exit_price - entry_price) / entry_price * direction
        trades.append(
            {
                "entry_time": df.loc[idx, "time"],
                "exit_time": df.loc[exit_idx, "time"],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            }
        )

    return pd.DataFrame(trades)


def backtest_long_short(
    df: pd.DataFrame,
    pred_col: str = "pred",
    price_col: str = "close",
    threshold: float = 0.0,
    cost_bps: float = 0.5,
    horizon: int = 3,
    enforce_hold: bool = True,
) -> BacktestResult:
    """Run a horizon-aware long/short backtest with next-bar execution."""
    data = df.sort_values("time").reset_index(drop=True).copy()

    data["fwd_ret_h"] = data[price_col].shift(-horizon) / data[price_col] - 1.0
    data = data.iloc[:-horizon].reset_index(drop=True)

    desired = np.where(data[pred_col] > threshold, 1, 0)
    desired = np.where(data[pred_col] < -threshold, -1, desired)
    data["desired_signal"] = desired

    position = np.zeros(len(data))
    hold_until = -1

    for t in range(len(data)):
        if t <= hold_until and enforce_hold:
            position[t] = position[t - 1] if t > 0 else 0
            continue

        if t == 0:
            position[t] = 0
            continue

        position[t] = data["desired_signal"].iloc[t - 1]
        if enforce_hold and position[t] != 0:
            hold_until = t + horizon - 1

    data["position"] = position

    cost_per_trade = cost_bps / 10000.0
    pos_prev = data["position"].shift(1).fillna(0)
    change = data["position"] - pos_prev
    trade_cost = np.where(change == 0, 0.0, 0.0)
    trade_cost += np.where((pos_prev == 0) & (data["position"] != 0), cost_per_trade, 0.0)
    trade_cost += np.where((pos_prev != 0) & (data["position"] == 0), cost_per_trade, 0.0)
    trade_cost += np.where((pos_prev == 1) & (data["position"] == -1), 2 * cost_per_trade, 0.0)
    trade_cost += np.where((pos_prev == -1) & (data["position"] == 1), 2 * cost_per_trade, 0.0)

    data["strategy_return"] = data["position"] * data["fwd_ret_h"] - trade_cost
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

    trades = _extract_trades_horizon(data, "position", horizon)

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
