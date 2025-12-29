"""Basic live metrics using executed trades (or backtest slice)."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def compute_trade_metrics(trades: Iterable[dict]) -> Dict[str, float]:
    df = pd.DataFrame(trades)
    if df.empty:
        return {
            "total_return": 0.0,
            "profit_factor": 0.0,
            "hit_rate": 0.0,
            "avg_trade_return": 0.0,
        }
    rets = df.get("pnl", pd.Series([], dtype=float))
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    total_return = float(rets.sum())
    profit_factor = float(wins.sum() / (np.abs(losses.sum()) + 1e-9)) if len(rets) else 0.0
    hit_rate = float((rets > 0).mean())
    avg_trade_return = float(rets.mean())
    return {
        "total_return": total_return,
        "profit_factor": profit_factor,
        "hit_rate": hit_rate,
        "avg_trade_return": avg_trade_return,
    }
