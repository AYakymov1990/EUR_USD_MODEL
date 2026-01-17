"""Lightweight explanations for signals (PL)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def build_explanation(features: Dict[str, float], y_hat: float) -> str:
    """Return short Polish explanation using key features."""
    parts: List[str] = []
    trend = features.get("ema_20_50_diff", 0.0)
    adx = features.get("adx_14", 0.0)
    h1 = features.get("h1_trend_flag", 0)
    parts.append(f"Prognoza modelu: {y_hat:.6f}")
    if trend > 0:
        parts.append("EMA20 powyżej EMA50 (lokalne bycze nachylenie)")
    else:
        parts.append("EMA20 poniżej EMA50 (lokalne niedźwiedzie nachylenie)")
    parts.append(f"ADX={adx:.1f}")
    parts.append(f"H1 trend-flag={h1}")
    return "; ".join(parts)


def confidence_from_pred(y_hat: float, pred_abs_p95: float | None = None) -> float:
    """Heuristic confidence based on prediction magnitude."""
    scale = pred_abs_p95 or max(abs(y_hat), 1e-6)
    return float(np.clip(abs(y_hat) / (scale + 1e-9), 0.0, 1.0))
