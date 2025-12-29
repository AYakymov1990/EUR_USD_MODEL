"""Signal generation and thresholding using selected config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtest import backtest_long_short_horizon


def load_selected_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"selected_config.json not found: {path}")
    return json.loads(path.read_text())


def apply_regime(row: pd.Series, regime: Optional[str]) -> bool:
    if regime in {None, "none"}:
        return True
    if regime == "adx":
        return row.get("adx_14", 0) >= 25
    if regime == "h1_align":
        return (row.get("h1_trend_flag", 0) == 1 and row.get("pred_dir", 1) > 0) or (
            row.get("h1_trend_flag", 0) == 0 and row.get("pred_dir", 1) < 0
        )
    if regime == "adx_and_h1":
        return apply_regime(row, "adx") and apply_regime(row, "h1_align")
    return True


def make_signal(row: pd.Series, pred: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    threshold = float(cfg.get("threshold", 0.0))
    polarity = float(cfg.get("polarity", 1.0))
    regime = cfg.get("regime")
    hold_bars = int(cfg.get("hold_bars", 3))
    cost_bps = float(cfg.get("cost_bps", 0.5))

    signed_pred = polarity * pred
    pred_dir = 1 if signed_pred > threshold else -1 if signed_pred < -threshold else 0
    row = row.copy()
    row["pred_dir"] = pred_dir
    regime_ok = apply_regime(row, regime)

    action = "none"
    if pred_dir == 1 and regime_ok:
        action = "long"
    elif pred_dir == -1 and regime_ok:
        action = "short"

    return {
        "y_hat": signed_pred,
        "threshold": threshold,
        "action": action,
        "regime_ok": regime_ok,
        "hold_bars": hold_bars,
        "cost_bps": cost_bps,
    }


def quick_backtest(df_slice: pd.DataFrame, pred: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to reuse canonical backtest for a recent slice (for metrics panel)."""
    bt = backtest_long_short_horizon(
        df_slice.assign(pred=pred),
        threshold=float(cfg.get("threshold", 0.0)),
        hold_bars=int(cfg.get("hold_bars", 3)),
        cost_bps=float(cfg.get("cost_bps", 0.5)),
        regime=cfg.get("regime"),
        sizing_mode=cfg.get("sizing_mode", "discrete"),
        target_ann_vol=cfg.get("target_ann_vol"),
    )
    return bt.metrics
