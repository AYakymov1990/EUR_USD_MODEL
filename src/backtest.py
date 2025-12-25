"""Backtesting utilities for model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


BARS_PER_YEAR = 252 * 24 * 4
BARS_PER_MONTH = 30 * 24 * 4


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity: pd.Series
    metrics: Dict[str, float]
    trades: pd.DataFrame
    debug: Dict[str, Any]

    @property
    def equity_curve(self) -> pd.Series:
        """Backwards-compatible alias for equity."""
        return self.equity


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _calc_costs(position: pd.Series, cost_bps: float) -> pd.Series:
    cost_per_trade = cost_bps / 10000.0
    pos_prev = position.shift(1).fillna(0)
    cost = pd.Series(0.0, index=position.index)

    cost += ((pos_prev == 0) & (position != 0)) * cost_per_trade
    cost += ((pos_prev != 0) & (position == 0)) * cost_per_trade
    cost += ((pos_prev == 1) & (position == -1)) * (2 * cost_per_trade)
    cost += ((pos_prev == -1) & (position == 1)) * (2 * cost_per_trade)
    return cost


def _extract_trades(df: pd.DataFrame, position_col: str) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    position = df[position_col].fillna(0)
    changes = position.diff().fillna(position)

    open_trade = None
    for idx in position.index:
        if changes.loc[idx] == 0:
            continue
        if open_trade is not None:
            open_trade["exit_time"] = df.loc[idx, "time"]
            open_trade["exit_price"] = df.loc[idx, "close"]
            open_trade["pnl"] = (
                (open_trade["exit_price"] - open_trade["entry_price"])
                / open_trade["entry_price"]
            ) * open_trade["direction"]
            trades.append(open_trade)
            open_trade = None
        if position.loc[idx] != 0:
            open_trade = {
                "entry_time": df.loc[idx, "time"],
                "entry_price": df.loc[idx, "close"],
                "direction": int(position.loc[idx]),
                "exit_time": None,
                "exit_price": None,
                "pnl": None,
            }

    return pd.DataFrame(trades)


def backtest_long_short(
    df: pd.DataFrame,
    pred_col: str = "pred",
    price_col: str = "close",
    threshold: float = 0.0,
    cost_bps: float = 0.5,
) -> BacktestResult:
    """Bar-by-bar backtest with next-bar execution and 1-bar returns."""
    data = df.sort_values("time").reset_index(drop=True).copy()
    data["ret_1"] = data[price_col].pct_change().fillna(0.0)

    desired = np.where(data[pred_col] > threshold, 1, 0)
    desired = np.where(data[pred_col] < -threshold, -1, desired)
    data["desired_signal"] = desired
    data["position"] = pd.Series(desired).shift(1).fillna(0).astype(int)

    cost = _calc_costs(data["position"], cost_bps)
    data["strategy_return"] = data["position"] * data["ret_1"] - cost
    data["equity"] = (1 + data["strategy_return"]).cumprod()

    total_return = data["equity"].iloc[-1] - 1.0 if len(data) else 0.0
    annualized_return = (
        (1 + total_return) ** (BARS_PER_YEAR / len(data)) - 1.0 if len(data) else 0.0
    )
    annualized_vol = data["strategy_return"].std() * np.sqrt(BARS_PER_YEAR) if len(data) else 0.0
    sharpe = float(annualized_return / annualized_vol) if annualized_vol > 0 else 0.0
    max_dd = _max_drawdown(data["equity"]) if len(data) else 0.0

    active = data["position"] != 0
    hit_rate = (
        (data.loc[active, "strategy_return"] > 0).mean() if active.any() else 0.0
    )

    trades = _extract_trades(data, "position")

    pos_counts = data["position"].value_counts().to_dict()
    desired_counts = pd.Series(data["desired_signal"]).value_counts().to_dict()
    pos_change_count = int((data["position"].diff().fillna(0) != 0).sum())

    n_bars = len(data)
    exposure_long = float((data["position"] == 1).mean()) if n_bars else 0.0
    exposure_short = float((data["position"] == -1).mean()) if n_bars else 0.0
    exposure_flat = float((data["position"] == 0).mean()) if n_bars else 0.0

    debug = {
        "n_bars": n_bars,
        "threshold": threshold,
        "cost_bps": cost_bps,
        "pos_unique_values": sorted(data["position"].unique().tolist()),
        "pos_change_count": pos_change_count,
        "exposure_long_pct": exposure_long,
        "exposure_short_pct": exposure_short,
        "exposure_flat_pct": exposure_flat,
        "pred_mean": float(data[pred_col].mean()),
        "pred_std": float(data[pred_col].std()),
        "pred_abs_p95": float(data[pred_col].abs().quantile(0.95)),
        "desired_signal_counts": {str(k): int(v) for k, v in desired_counts.items()},
        "executed_pos_counts": {str(k): int(v) for k, v in pos_counts.items()},
    }

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
        equity=data["equity"],
        metrics=metrics,
        trades=trades,
        debug=debug,
    )


def backtest_long_short_horizon(
    df: pd.DataFrame,
    pred_col: str = "pred",
    price_col: str = "close",
    threshold: float = 0.0,
    cost_bps: float = 0.5,
    hold_bars: int = 3,
    regime: str | None = None,
    adx_col: str = "adx_14",
    adx_min: float = 25.0,
    h1_trend_col: str = "h1_trend_flag",
    sizing_mode: str = "discrete",
    use_position_sizing: bool = False,
    leverage: float = 1.0,
    pred_scale_mode: str = "p95",
    pred_scale: float | None = None,
    pred_scale_q: float = 0.95,
    target_ann_vol: float | None = None,
    vol_lookback: int = 96,
    max_leverage: float = 3.0,
    eps: float = 1e-12,
) -> BacktestResult:
    """Horizon-aligned backtest for y = close[t+3] / close[t] - 1.

    Uses bar-by-bar returns with fixed holding window, no overlapping trades.
    """
    data = df.sort_values("time").reset_index(drop=True).copy()
    n = len(data)
    cost_per_trade = cost_bps / 10000.0

    trades: list[dict[str, Any]] = []
    ret_1 = data[price_col].pct_change().fillna(0.0).values
    log_ret = np.log(data[price_col]).diff().fillna(0.0)
    rolling_vol = log_ret.rolling(vol_lookback).std().bfill()
    ann_vol_price = rolling_vol * np.sqrt(BARS_PER_YEAR)

    position = np.zeros(n, dtype=float)

    abs_pred = data[pred_col].abs()
    pred_mean = float(data[pred_col].mean())
    pred_std = float(data[pred_col].std())
    pred_abs_p90 = float(abs_pred.quantile(0.90))
    pred_abs_p95 = float(abs_pred.quantile(0.95))

    signal_counts = {"long": 0, "short": 0, "none": 0}

    if sizing_mode not in {"discrete", "continuous"}:
        raise ValueError("sizing_mode must be 'discrete' or 'continuous'")
    if sizing_mode == "continuous":
        use_position_sizing = True
    else:
        use_position_sizing = False

    if pred_scale is None:
        scale = float(np.quantile(abs_pred, pred_scale_q))
    else:
        scale = float(pred_scale)
    scale = scale if scale > 0 else 1.0

    i = 0
    while i + 1 + hold_bars < n:
        pred = data[pred_col].iloc[i]
        if abs(pred) < threshold:
            direction = 0
            size = 0.0
        else:
            direction = 1 if pred > 0 else -1
            if use_position_sizing:
                raw = pred / (scale + eps)
                base_size = float(np.clip(abs(raw), 0.0, 1.0))
                if target_ann_vol is not None:
                    lev_i = float(
                        np.clip(target_ann_vol / (ann_vol_price.iloc[i] + eps), 0.0, max_leverage)
                    )
                else:
                    lev_i = leverage
                size = base_size * lev_i
            else:
                size = 1.0

        if direction == 0:
            signal_counts["none"] += 1
            i += 1
            continue

        if regime in {"adx", "adx_and_h1"}:
            if data[adx_col].iloc[i] < adx_min:
                signal_counts["none"] += 1
                i += 1
                continue
        if regime in {"h1_align", "adx_and_h1"}:
            h1_flag = data[h1_trend_col].iloc[i]
            if (direction == 1 and h1_flag != 1) or (direction == -1 and h1_flag != 0):
                signal_counts["none"] += 1
                i += 1
                continue

        if direction == 1:
            signal_counts["long"] += 1
        else:
            signal_counts["short"] += 1

        entry_idx = i + 1
        exit_idx = entry_idx + hold_bars
        entry_price = data[price_col].iloc[entry_idx]
        exit_price = data[price_col].iloc[exit_idx]

        position[entry_idx:exit_idx] = direction * size

        trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_time": data["time"].iloc[entry_idx],
                "exit_time": data["time"].iloc[exit_idx],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": size,
                "trade_return": 0.0,
                "pnl": 0.0,
            }
        )

        i = exit_idx + 1

    pos_series = pd.Series(position)
    turnover = (pos_series - pos_series.shift(1).fillna(0.0)).abs()
    cost_series = turnover * (cost_bps / 10000.0)

    strategy_ret = pos_series.values * ret_1 - cost_series.values
    equity = pd.Series((1 + strategy_ret).cumprod(), index=data.index, dtype=float)

    if trades:
        for t in trades:
            start = t["entry_idx"]
            end = min(t["exit_idx"] + 1, n)
            trade_return = float(np.prod(1 + strategy_ret[start:end]) - 1.0)
            t["trade_return"] = trade_return
            t["pnl"] = trade_return

    trades_df = pd.DataFrame(trades)
    trade_returns = trades_df["trade_return"] if not trades_df.empty else pd.Series([], dtype=float)
    total_return = equity.iloc[-1] - 1.0 if len(equity) else 0.0
    first_time = pd.to_datetime(data["time"].iloc[0])
    last_time = pd.to_datetime(data["time"].iloc[-1])
    years = max(
        (last_time - first_time).total_seconds() / (365.25 * 24 * 3600), eps
    )
    annualized_return = float(np.exp(np.log1p(total_return) / years) - 1.0)
    annualized_vol = (
        np.std(strategy_ret, ddof=0) * np.sqrt(BARS_PER_YEAR) if len(strategy_ret) else 0.0
    )
    sharpe = (
        float(np.mean(strategy_ret) / (np.std(strategy_ret, ddof=0) + eps)) * np.sqrt(BARS_PER_YEAR)
        if len(strategy_ret)
        else 0.0
    )
    max_dd = _max_drawdown(equity) if len(equity) else 0.0
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    denom = len(wins) + len(losses)
    hit_rate = float(len(wins) / denom) if denom > 0 else 0.0
    avg_trade_return = float(trade_returns.mean()) if len(trade_returns) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = (
        float(np.abs(losses).mean()) if len(losses) else 0.0
    )
    payoff_ratio = avg_win / (avg_loss + eps) if avg_loss > 0 else 0.0
    profit_factor = (
        float(wins.sum()) / (np.abs(losses.sum()) + eps) if len(trade_returns) else 0.0
    )
    monthly_return_est = float(np.exp(np.log1p(total_return) / (years * 12)) - 1.0)

    avg_holding_bars = float(hold_bars) if trades else 0.0

    nonzero_pos_bars = int(np.sum(pos_series.values != 0))
    nonzero_stratret_bars = int(np.sum(strategy_ret != 0))
    trade_pnl_sum = float(trade_returns.sum()) if len(trade_returns) else 0.0
    trade_geom_total_return = float(np.prod(1 + trade_returns) - 1.0) if len(trade_returns) else 0.0
    equity_last = float(equity.iloc[-1]) if len(equity) else 1.0
    if len(trade_returns) and (trade_returns <= -1.0 + eps).any():
        trade_log_sum = float("nan")
        equity_log = float("nan")
        consistency_abs = float("nan")
    else:
        trade_log_sum = float(np.sum(np.log1p(trade_returns))) if len(trade_returns) else 0.0
        equity_log = float(np.log(equity_last)) if equity_last > 0 else float("nan")
        consistency_abs = abs(trade_log_sum - equity_log)

    debug = {
        "n_bars": len(data),
        "hold_bars": hold_bars,
        "threshold": threshold,
        "regime": regime,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_abs_p90": pred_abs_p90,
        "pred_abs_p95": pred_abs_p95,
        "signal_counts": signal_counts,
        "trade_count": float(len(trades)),
        "avg_holding_bars": avg_holding_bars,
        "equity_last": equity_last,
        "strat_ret_sum": float(np.sum(strategy_ret)) if len(strategy_ret) else 0.0,
        "nonzero_pos_bars": nonzero_pos_bars,
        "nonzero_stratret_bars": nonzero_stratret_bars,
        "trade_pnl_sum": trade_pnl_sum,
        "trade_geom_total_return": trade_geom_total_return,
        "trade_log_sum": trade_log_sum,
        "equity_log": equity_log,
        "consistency_abs": consistency_abs,
        "use_position_sizing": use_position_sizing,
        "leverage": leverage,
        "pred_scale_mode": pred_scale_mode,
        "bars_per_year": BARS_PER_YEAR,
        "bars_per_month": BARS_PER_MONTH,
    }

    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_vol": float(annualized_vol) if not np.isnan(annualized_vol) else 0.0,
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "hit_rate": float(hit_rate),
        "trade_count": float(len(trades_df)),
        "avg_trade_return": float(avg_trade_return),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff_ratio": float(payoff_ratio),
        "profit_factor": float(profit_factor),
        "monthly_return_est": float(monthly_return_est),
    }

    if len(trades) > 0 and consistency_abs > 1e-5:
        debug["warning"] = "trade_count>0 but trade/equity logs mismatch"

    return BacktestResult(
        equity=equity,
        metrics=metrics,
        trades=trades_df,
        debug=debug,
    )
