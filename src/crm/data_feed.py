"""Data feed layer: demo replay or live OANDA fetch (placeholder)."""

from __future__ import annotations

from datetime import datetime
from typing import Generator, Iterable, Optional

import pandas as pd

from .config import CRMConfig


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
    oanda_client: Optional[object] = None,
) -> Iterable[dict]:
    """
    Placeholder for live OANDA candle fetching.

    Returns an empty iterable in demo mode; caller should provide a client in live mode.
    """
    if cfg.demo_mode or not cfg.allow_live:
        return []
    if oanda_client is None:
        return []
    # The concrete implementation should call OANDA /candles with cfg.instrument/cfg.granularity.
    return []


def parse_time(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(ts)
