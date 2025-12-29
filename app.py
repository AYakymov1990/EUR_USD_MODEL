import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.crm.config import CRMConfig, load_config
from src.crm.data_feed import demo_feed
from src.crm.explain import build_explanation, confidence_from_pred
from src.crm.features_adapter import build_feature_row
from src.crm.inference import load_artifacts, run_inference
from src.crm.oanda_executor import fetch_account, place_market_order
from src.crm.signals import load_selected_config, make_signal
from src.crm.storage import (
    ensure_schema,
    fetch_recent_signals,
    get_connection,
    log_action,
    log_order_event,
    log_signal,
)


@st.cache_resource
def get_cfg() -> CRMConfig:
    return load_config()


@st.cache_resource
def get_selected(cfg: CRMConfig) -> dict:
    try:
        return load_selected_config(cfg.selected_config_path)
    except Exception:
        return {}


@st.cache_resource
def get_db(cfg: CRMConfig):
    conn = get_connection(cfg.sqlite_path)
    ensure_schema(conn)
    return conn


def load_demo_rows(cfg: CRMConfig) -> list:
    return list(demo_feed(cfg))


def main() -> None:
    cfg = get_cfg()
    sel = get_selected(cfg)
    conn = get_db(cfg)

    st.set_page_config(page_title="Trader CRM", layout="wide")
    st.title("Trader CRM (demo)")
    st.caption("Demo mode by default. Live requires .env with OANDA credentials.")

    # Sidebar config
    st.sidebar.subheader("Config")
    st.sidebar.write(cfg)
    artifacts_dir = Path(cfg.artifacts_dir)
    model_path = artifacts_dir / "model.pt"
    scaler_path = artifacts_dir / "scaler.pkl"

    # Demo data cache
    if "demo_rows" not in st.session_state:
        st.session_state["demo_rows"] = load_demo_rows(cfg)
        st.session_state["demo_idx"] = 0

    # Load artifacts lazily
    model = scaler = None
    artifacts_ok = False
    if model_path.exists() and scaler_path.exists():
        try:
            model, scaler = load_artifacts(model_path, scaler_path)
            artifacts_ok = True
        except Exception as exc:  # pragma: no cover
            st.sidebar.error(f"Не удалось загрузить артефакты: {exc}")
    else:
        st.sidebar.warning("Модель/скейлер не найдены. Сигналы будут фиктивными.")

    # Account snapshot
    account_col, signal_col = st.columns(2)
    with account_col:
        st.subheader("Аккаунт")
        acct = fetch_account(cfg)
        st.json(acct)

    # Signal generation
    if st.button("Сгенерировать следующий сигнал (demo feed)"):
        rows = st.session_state["demo_rows"]
        if not rows:
            st.error("Нет данных для демо.")
        else:
            idx = st.session_state.get("demo_idx", 0) % len(rows)
            row = rows[idx]
            st.session_state["demo_idx"] = idx + 1
            feature_df = build_feature_row(row)
            if artifacts_ok:
                pred_info = run_inference(model, scaler, feature_df.values)
                y_hat = pred_info["y_hat_scalar"]
            else:
                y_hat = float(row.get("target", 0.0))
            signal = make_signal(row, y_hat, sel)
            conf = confidence_from_pred(y_hat, row.get("pred_abs_p95"))
            ts = row.get("time", datetime.utcnow()).isoformat()
            log_signal(conn, ts, y_hat, signal["action"], conf, signal)
            st.success(f"Сигнал {signal['action']} @ {y_hat:.6f}")
            st.session_state["last_signal"] = {
                "ts": ts,
                "y_hat": y_hat,
                "action": signal["action"],
                "explanation": build_explanation(row, y_hat),
                "confidence": conf,
            }

    # UI display
    with signal_col:
        st.subheader("Последний сигнал")
        last = st.session_state.get("last_signal")
        if last:
            st.write(last["ts"])
            st.metric("Действие", last["action"], f"{last['y_hat']:.6f}")
            st.progress(min(max(last["confidence"], 0), 1.0))
            st.text(last["explanation"])
        else:
            st.info("Сигналов пока нет.")

    # Actions
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Подтвердить LONG"):
            ts = datetime.utcnow().isoformat()
            log_action(conn, ts, "long", "user_confirmed", {})
            ok, resp = place_market_order(cfg, units=1)
            log_order_event(conn, ts, "long", 1, "sent" if ok else "skipped", resp)
            st.success(f"LONG отправлен (demo={cfg.demo_mode})")
    with action_col2:
        if st.button("Подтвердить SHORT"):
            ts = datetime.utcnow().isoformat()
            log_action(conn, ts, "short", "user_confirmed", {})
            ok, resp = place_market_order(cfg, units=-1)
            log_order_event(conn, ts, "short", -1, "sent" if ok else "skipped", resp)
            st.success(f"SHORT отправлен (demo={cfg.demo_mode})")

    # Recent signals table
    st.subheader("Журнал сигналов (SQLite)")
    signals = fetch_recent_signals(conn, limit=30)
    if signals:
        df_sig = pd.DataFrame(signals)
        st.dataframe(df_sig)
    else:
        st.info("Журнал пуст.")


if __name__ == "__main__":
    main()
