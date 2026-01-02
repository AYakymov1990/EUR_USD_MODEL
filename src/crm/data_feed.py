"""Data feed layer: demo replay or live OANDA fetch (placeholder)."""

from __future__ import annotations

from datetime import datetime
from typing import Generator, Iterable, Optional, Tuple

import pandas as pd
import requests

from .config import CRMConfig
from .oanda_executor import _resolve_domain


def demo_feed(cfg: CRMConfig) -> Generator[pd.Series, None, None]:
    """
    Replay rows from prepared features parquet for demo mode.

    Yields pandas Series rows with feature columns and time.
    """
    if not cfg.features_path.exists():
        raise FileNotFoundError(f"Demo features file not found: {cfg.features_path}")
    df = pd.read_parquet(cfg.features_path)
    df = df.sort_values("time").reset_index(drop=True)
    if cfg.max_demo_rows:
        df = df.tail(cfg.max_demo_rows)
    for _, row in df.iterrows():
        yield row


def live_candles_from_oanda(
    cfg: CRMConfig,
    instrument: str,
    granularity: str,
    count: int = 300,
    price: str = "M",
) -> pd.DataFrame:
    """
    Fetch candles from OANDA. Returns empty DF on failure (caller handles graceful degradation).
    """
    if cfg.demo_mode or not cfg.allow_live:
        return pd.DataFrame()

    domain = _resolve_domain(cfg)
    url = f"https://{domain}/v3/instruments/{instrument}/candles"
    params = {
        "granularity": granularity,
        "count": count,
        "price": price,
    }
    headers = {
        "Authorization": f"Bearer {cfg.oanda_api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=cfg.timeout)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame()

    data = resp.json()
    candles = data.get("candles", [])
    rows: list[dict[str, object]] = []
    for c in candles:
        if not c.get("complete"):
            continue
        mid = c.get("mid", {})
        rows.append(
            {
                "time": pd.to_datetime(c.get("time")),
                "open": float(mid.get("o")),
                "high": float(mid.get("h")),
                "low": float(mid.get("l")),
                "close": float(mid.get("c")),
                "volume": int(c.get("volume", 0)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("time").reset_index(drop=True)
    return df


def get_latest_live_window(
    cfg: CRMConfig,
    instrument: Optional[str] = None,
    m15_count: int = 300,
    h1_count: int = 300,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch recent M15 and H1 windows for feature computation."""
    inst = instrument or cfg.instrument
    df_m15 = live_candles_from_oanda(cfg, inst, "M15", count=m15_count)
    df_h1 = live_candles_from_oanda(cfg, inst, "H1", count=h1_count)
    if df_m15.empty or df_h1.empty:
        return None, None
    return df_m15, df_h1


def parse_time(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(ts)
