"""SQLite storage for audit trail and metrics snapshots."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    y_hat REAL,
    action TEXT,
    regime_ok INTEGER,
    confidence REAL,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    action TEXT,
    reason TEXT,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    direction TEXT,
    size REAL,
    status TEXT,
    response TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    direction TEXT,
    size REAL,
    price REAL,
    pnl REAL,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    equity REAL,
    profit_factor REAL,
    sharpe REAL,
    max_drawdown REAL,
    payload TEXT
);
"""


def get_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _to_serializable(obj: Any) -> Any:
    """Convert common non-JSON types to serializable primitives/strings."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    # pandas.Timestamp also exposes isoformat; avoid hard dependency by duck-typing
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _dumps_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(_to_serializable(payload))


def log_signal(conn: sqlite3.Connection, ts: str, y_hat: float, action: str, confidence: float, payload: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO signals(ts, y_hat, action, regime_ok, confidence, payload) VALUES (?, ?, ?, ?, ?, ?)",
        (ts, y_hat, action, int(payload.get("regime_ok", 1)), confidence, _dumps_payload(payload)),
    )
    conn.commit()


def log_action(conn: sqlite3.Connection, ts: str, action: str, reason: str, payload: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO actions(ts, action, reason, payload) VALUES (?, ?, ?, ?)",
        (ts, action, reason, _dumps_payload(payload)),
    )
    conn.commit()


def log_order_event(conn: sqlite3.Connection, ts: str, direction: str, size: float, status: str, response: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO orders(ts, direction, size, status, response) VALUES (?, ?, ?, ?, ?)",
        (ts, direction, size, status, _dumps_payload(response)),
    )
    conn.commit()


def fetch_recent_signals(conn: sqlite3.Connection, limit: int = 20) -> List[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
    return cur.fetchall()


def get_last_signal_time(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.execute("SELECT ts FROM signals ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    return row["ts"] if row else None
