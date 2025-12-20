"""OANDA v20 REST API client wrapper."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class OandaConfig:
    """Configuration for OANDA API access."""

    api_key: str
    account_id: str
    environment: str
    base_url_practice: str
    base_url_live: str

    @property
    def base_url(self) -> str:
        environment = self.environment.lower()
        if environment == "practice":
            return self.base_url_practice
        return self.base_url_live

    @classmethod
    def from_json(cls, path: str | Path) -> "OandaConfig":
        """Load config from a JSON file."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(**raw)


class OandaClient:
    """Minimal OANDA v20 REST client for accounts and candles."""

    def __init__(self, config: OandaConfig, timeout: int = 10) -> None:
        self.config = config
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _to_oanda_time(dt: datetime) -> str:
        """Convert a datetime to OANDA ISO8601 format with Z (UTC)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def get_account_details(self) -> dict[str, Any]:
        """Fetch account details using GET /v3/accounts/{accountID}."""
        url = f"{self.config.base_url}/v3/accounts/{self.config.account_id}"
        try:
            response = requests.get(url, headers=self._headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Failed to fetch account details from OANDA.")
            raise RuntimeError(f"OANDA request failed: {exc}") from exc
        return response.json()

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        count: Optional[int] = None,
        from_time: str | None = None,
        to_time: str | None = None,
        price: str = "M",
        smooth: bool = False,
        include_first: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical candles using GET /v3/instruments/{instrument}/candles.

        Args:
            instrument: OANDA instrument, e.g. "EUR_USD".
            granularity: Candle granularity, e.g. "M15" or "H1".
            count: Number of candles to request (alternative to from/to).
            from_time: ISO8601 start time.
            to_time: ISO8601 end time.
            price: Price component, default "M" for midpoint.
            smooth: Whether to return smoothed candles.
            include_first: Whether to include the first candle for the from-time.
        """
        if count is None and (from_time is None or to_time is None):
            raise ValueError("Provide either count or both from_time and to_time")

        params: dict[str, Any] = {
            "granularity": granularity,
            "price": price,
            "smooth": "true" if smooth else "false",
        }
        if from_time is not None:
            params["from"] = from_time
        if to_time is not None:
            params["to"] = to_time
        if from_time is not None and to_time is not None:
            params["includeFirst"] = "true" if include_first else "false"
        if count is not None:
            params["count"] = count

        url = f"{self.config.base_url}/v3/instruments/{instrument}/candles"
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Failed to fetch candles from OANDA.")
            raise RuntimeError(f"OANDA request failed: {exc}") from exc

        data = response.json()

        candles = data.get("candles", [])
        rows: list[dict[str, Any]] = []
        for candle in candles:
            if not candle.get("complete", False):
                continue
            mid = candle.get("mid", {})
            rows.append(
                {
                    "time": pd.to_datetime(candle.get("time")),
                    "open": float(mid.get("o")),
                    "high": float(mid.get("h")),
                    "low": float(mid.get("l")),
                    "close": float(mid.get("c")),
                    "volume": int(candle.get("volume")),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("No completed candles returned for %s %s.", instrument, granularity)
            return df

        df = df.sort_values("time").reset_index(drop=True)
        return df

    def get_candles_range(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: datetime,
        price: str = "M",
        max_per_request: int = 5000,
        sleep_sec: float = 0.2,
    ) -> pd.DataFrame:
        """Fetch candles in [start, end) using multiple requests (pagination by time)."""
        start_utc = (
            start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
        )
        end_utc = (
            end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
        )

        current_start = start_utc
        chunks: list[pd.DataFrame] = []

        while current_start < end_utc:
            from_str = self._to_oanda_time(current_start)
            logger.info(
                "Fetching %s %s from %s (max %s)",
                instrument,
                granularity,
                from_str,
                max_per_request,
            )
            df_chunk = self.get_candles(
                instrument=instrument,
                granularity=granularity,
                count=max_per_request,
                from_time=from_str,
                price=price,
            )

            if df_chunk.empty:
                logger.warning("Empty candle batch returned for %s %s.", instrument, granularity)
                break

            last_time = df_chunk["time"].max()
            if pd.isna(last_time):
                logger.warning("No valid time in candle batch for %s %s.", instrument, granularity)
                break

            last_dt = last_time.to_pydatetime()
            if last_dt <= current_start:
                logger.warning(
                    "Candle time did not advance (last=%s, current=%s). Stopping.",
                    last_dt,
                    current_start,
                )
                break

            chunks.append(df_chunk)
            current_start = last_dt + timedelta(seconds=1)
            time.sleep(sleep_sec)

        if not chunks:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df_all = pd.concat(chunks, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        df_all = df_all[(df_all["time"] >= start_utc) & (df_all["time"] < end_utc)].reset_index(drop=True)
        return df_all
