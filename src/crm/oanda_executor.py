"""Lightweight OANDA executor (practice by default)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import requests

from .config import CRMConfig


def _headers(cfg: CRMConfig) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.oanda_api_key}",
    }


def place_market_order(
    cfg: CRMConfig,
    units: float,
    client_order_id: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    if cfg.demo_mode or not cfg.allow_live:
        return False, {"message": "Demo mode: order not sent"}
    url = f"https://api-{cfg.oanda_env}.oanda.com/v3/accounts/{cfg.oanda_account_id}/orders"
    payload = {
        "order": {
            "type": "MARKET",
            "instrument": cfg.instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
        }
    }
    if client_order_id:
        payload["order"]["clientExtensions"] = {"id": client_order_id}
    try:
        resp = requests.post(url, headers=_headers(cfg), data=json.dumps(payload), timeout=cfg.timeout)
        return resp.ok, {"status": resp.status_code, "text": resp.text}
    except Exception as exc:
        return False, {"error": str(exc)}


def fetch_account(cfg: CRMConfig) -> Dict[str, Any]:
    if cfg.demo_mode or not cfg.allow_live:
        return {"demo": True}
    url = f"https://api-{cfg.oanda_env}.oanda.com/v3/accounts/{cfg.oanda_account_id}/summary"
    try:
        resp = requests.get(url, headers=_headers(cfg), timeout=cfg.timeout)
        if resp.ok:
            return resp.json()
        return {"status": resp.status_code, "text": resp.text}
    except Exception as exc:
        return {"error": str(exc)}


def close_position(cfg: CRMConfig, units: float) -> Tuple[bool, Dict[str, Any]]:
    """Close by sending reverse market order."""
    return place_market_order(cfg, units=-units)
