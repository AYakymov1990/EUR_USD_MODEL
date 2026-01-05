"""FastAPI server exposing Trader CRM backend endpoints."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.crm.config import CRMConfig, load_config
from src.crm.data_feed import demo_feed, get_latest_live_window
from src.crm.explain import build_explanation, confidence_from_pred
from src.crm.features_adapter import build_feature_row, build_live_feature_row
from src.crm.inference import load_artifacts, run_inference
from src.crm.notifications import send_email
from src.crm.oanda_executor import fetch_account, place_market_order
from src.crm.signals import load_selected_config, make_signal
from src.crm.storage import (
    ensure_schema,
    fetch_recent_signals,
    get_connection,
    get_last_signal_time,
    log_action,
    log_order_event,
    log_signal,
)


class OrderRequest(BaseModel):
    direction: str
    units: int = 1


class SignalResponse(BaseModel):
    ts: str
    action: str
    y_hat: float
    confidence: float
    explanation: str
    meta: Dict[str, Any] = {}


def _load_demo_rows(cfg: CRMConfig) -> List[pd.Series]:
    return list(demo_feed(cfg))


class CRMService:
    """Stateful helper mirroring Streamlit behaviors (demo replay, logging)."""

    def __init__(self, cfg: CRMConfig) -> None:
        self.cfg = cfg
        self.conn = get_connection(cfg.sqlite_path)
        ensure_schema(self.conn)
        self.sel = load_selected_config(cfg.selected_config_path) or {}
        self.demo_rows = _load_demo_rows(cfg) if cfg.demo_mode else []
        self.demo_idx = 0
        self.demo_lock = threading.Lock()
        self.model = None
        self.scaler = None
        self.artifacts_ok = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_path = self.cfg.artifacts_dir / "model.pt"
        scaler_path = self.cfg.artifacts_dir / "scaler.pkl"
        if model_path.exists() and scaler_path.exists():
            try:
                self.model, self.scaler = load_artifacts(model_path, scaler_path)
                self.artifacts_ok = True
            except Exception:
                self.artifacts_ok = False

    def _next_demo_row(self) -> Optional[pd.Series]:
        if not self.demo_rows:
            return None
        with self.demo_lock:
            row = self.demo_rows[self.demo_idx % len(self.demo_rows)]
            self.demo_idx += 1
        return row

    def _send_signal_email(self, signal: dict, meta: dict, y_hat: float, ts: str) -> None:
        if not self.cfg.allow_email:
            return
        direction = signal.get("action")
        if direction not in {"long", "short"}:
            return
        subject = f"Trader CRM: сигнал {direction.upper()} @ {ts}"
        body = (
            f"Время: {ts}\n"
            f"Действие: {direction}\n"
            f"Прогноз y_hat: {y_hat}\n"
            f"Причина: {signal.get('reason','')}\n"
            f"Режим: {signal.get('regime','')}\n"
            f"Цена: {meta.get('close')}"
        )
        send_email(self.cfg, subject, body)

    def generate_signal(self) -> SignalResponse:
        cfg = self.cfg
        now = datetime.utcnow()

        if cfg.demo_mode:
            row = self._next_demo_row()
            if row is None:
                raise HTTPException(status_code=400, detail="Нет демо-данных")
            feature_df = build_feature_row(row)
            y_hat = 0.0
            if self.artifacts_ok:
                pred_info = run_inference(self.model, self.scaler, feature_df.values)
                y_hat = pred_info["y_hat_scalar"]
            else:
                y_hat = float(row.get("target", 0.0))
            sel_current = dict(self.sel)
            signal = make_signal(row, y_hat, sel_current)
            conf = confidence_from_pred(y_hat, row.get("pred_abs_p95"))
            ts = row.get("time", now).isoformat()
            log_signal(self.conn, ts, y_hat, signal["action"], conf, signal)
            explanation = build_explanation(row, y_hat)
            last_payload = {**signal, "explanation": explanation}
            return SignalResponse(
                ts=ts,
                action=signal["action"],
                y_hat=y_hat,
                confidence=conf,
                explanation=explanation,
                meta=last_payload,
            )

        df_m15, df_h1 = get_latest_live_window(cfg)
        if df_m15 is None or df_h1 is None or df_m15.empty or df_h1.empty:
            raise HTTPException(status_code=502, detail="Не удалось получить свечи OANDA")
        feature_df, last_row, meta = build_live_feature_row(df_m15, df_h1)
        y_hat = 0.0
        if self.artifacts_ok:
            pred_info = run_inference(self.model, self.scaler, feature_df.values)
            y_hat = pred_info["y_hat_scalar"]
        else:
            y_hat = float(last_row.get("target", 0.0))
        sel_current = dict(self.sel)
        signal = make_signal(last_row, y_hat, sel_current)
        conf = confidence_from_pred(y_hat)
        ts_val = meta.get("time") or now
        ts = ts_val.isoformat()
        last_logged = get_last_signal_time(self.conn)
        if not last_logged or last_logged != ts:
            log_signal(self.conn, ts, y_hat, signal["action"], conf, {**signal, **meta})
            self._send_signal_email(signal, meta, y_hat, ts)
        explanation = build_explanation(last_row, y_hat)
        return SignalResponse(
            ts=ts,
            action=signal["action"],
            y_hat=y_hat,
            confidence=conf,
            explanation=explanation,
            meta={**signal, **meta},
        )


cfg = load_config()
svc = CRMService(cfg)
app = FastAPI(title="Trader CRM API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "demo": cfg.demo_mode}


@app.get("/config")
def get_config() -> dict:
    return {
        "demo_mode": cfg.demo_mode,
        "signal_mode": cfg.signal_mode,
        "instrument": cfg.instrument,
        "granularity": cfg.granularity,
        "allow_live": cfg.allow_live,
    }


@app.get("/account")
def account() -> dict:
    return fetch_account(cfg)


@app.get("/news")
def news(limit: int = 5) -> dict:
    return {"items": fetch_recent_news(cfg, limit)}


def fetch_recent_news(cfg: CRMConfig, limit: int = 5) -> List[dict]:
    """News fetch mirroring Streamlit flow with fallback to empty list."""
    api_key = cfg.news_api_key
    if not api_key:
        return []
    import requests

    url = "https://newsapi.org/v2/top-headlines"
    params = {"q": "EURUSD", "apiKey": api_key, "language": "en", "pageSize": limit, "country": "us"}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        articles = data.get("articles", []) or []
        if articles:
            return articles
    except Exception:
        pass

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "EURUSD",
            "apiKey": api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
        }
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data.get("articles", []) or []
    except Exception:
        return []


@app.get("/signals/recent")
def signals_recent(limit: int = 50) -> dict:
    rows = fetch_recent_signals(svc.conn, limit=limit)
    result = []
    for r in rows:
        payload = r["payload"]
        try:
            payload_obj = payload if isinstance(payload, dict) else json.loads(payload)
        except Exception:
            payload_obj = {}
        result.append(
            {
                "ts": r["ts"],
                "y_hat": r["y_hat"],
                "action": r["action"],
                "confidence": r["confidence"],
                "payload": payload_obj,
            }
        )
    return {"items": result}


@app.post("/signals/generate", response_model=SignalResponse)
def generate_signal() -> SignalResponse:
    return svc.generate_signal()


@app.post("/orders/market")
def create_order(req: OrderRequest) -> dict:
    direction = req.direction.lower()
    if direction not in {"long", "short"}:
        raise HTTPException(status_code=400, detail="direction must be long|short")
    units = req.units if direction == "long" else -req.units
    ts = datetime.utcnow().isoformat()
    log_action(svc.conn, ts, direction, "user_submit", {"units": units})
    ok, resp = place_market_order(cfg, units=units)
    log_order_event(svc.conn, ts, direction, units, "sent" if ok else "skipped", resp)
    return {"ok": ok, "response": resp}
