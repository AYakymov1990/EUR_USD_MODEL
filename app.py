import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

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


def _get_query_params() -> dict:
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()


def _set_query_params(params: dict) -> None:
    try:
        st.query_params.clear()
        st.query_params.update(params)
    except Exception:
        st.experimental_set_query_params(**params)


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


def send_signal_email(cfg: CRMConfig, signal: dict, meta: dict, y_hat: float, ts: str) -> None:
    if not cfg.allow_email:
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
        f"H1 trend: {meta.get('h1_trend_flag')} ADX: {meta.get('adx_val')}\n"
        f"Цена: {meta.get('close')}"
    )
    ok, err = send_email(cfg, subject, body)
    if not ok:
        st.warning(f"Не удалось отправить email: {err}")


def main() -> None:
    cfg = get_cfg()
    sel = get_selected(cfg)
    conn = get_db(cfg)

    st.set_page_config(page_title="Trader CRM", layout="wide")
    st.title("Trader CRM")
    mode_label = "LIVE (practice)" if not cfg.demo_mode else "DEMO"
    st.caption(f"Mode: {mode_label}. Live требует .env с ключами OANDA.")

    # Sidebar config
    st.sidebar.subheader("Config")
    st.sidebar.write(cfg)
    artifacts_dir = Path(cfg.artifacts_dir)
    model_path = artifacts_dir / "model.pt"
    scaler_path = artifacts_dir / "scaler.pkl"

    # Demo data cache
    if cfg.demo_mode and "demo_rows" not in st.session_state:
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
    if cfg.demo_mode:
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
    else:
        qp = _get_query_params()
        auto_default = qp.get("auto", ["0"])[0] in {"1", "true", "yes"}
        auto_mode = st.sidebar.checkbox("Авто-режим (каждые 15м)", value=auto_default)
        if auto_mode != auto_default:
            qp["auto"] = "1" if auto_mode else "0"
            _set_query_params(qp)

        # Автообновление: раз в 60 сек триггерим rerun, если есть autorefresh
        st_autorefresh = getattr(st, "autorefresh", None)
        if st_autorefresh:
            st_autorefresh(interval=60_000, key="auto_refresh")
        elif auto_mode:
            # Fallback: meta refresh в <head> — более надёжно, чем JS в теле
            st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

        now = datetime.utcnow()
        last_run = st.session_state.get("last_auto_run")
        ready_for_auto = not last_run or (now - last_run) > timedelta(minutes=1)

        manual_clicked = st.button("Получить live сигнал")

        should_run = manual_clicked or (auto_mode and ready_for_auto)
        if auto_mode and should_run:
            st.session_state["last_auto_run"] = now

        if should_run:
            df_m15, df_h1 = get_latest_live_window(cfg)
            if df_m15 is None or df_h1 is None or df_m15.empty or df_h1.empty:
                st.error("Не удалось получить свечи OANDA. Проверьте сеть/ключи.")
            else:
                try:
                    feature_df, last_row, meta = build_live_feature_row(df_m15, df_h1)
                except Exception as exc:
                    st.error(f"Ошибка подготовки признаков: {exc}")
                else:
                    if artifacts_ok:
                        pred_info = run_inference(model, scaler, feature_df.values)
                        y_hat = pred_info["y_hat_scalar"]
                    else:
                        y_hat = float(last_row.get("target", 0.0))
                    signal = make_signal(last_row, y_hat, sel)
                    conf = confidence_from_pred(y_hat)
                    ts_val = meta.get("time") or datetime.utcnow()
                    ts = ts_val.isoformat()
                    last_logged = get_last_signal_time(conn)
                    if last_logged and last_logged == ts:
                        if manual_clicked:
                            st.info("Сигнал для этого бара уже зафиксирован.")
                    else:
                        log_signal(conn, ts, y_hat, signal["action"], conf, {**signal, **meta})
                        send_signal_email(cfg, signal, meta, y_hat, ts)
                        st.success(f"Сигнал {signal['action']} @ {y_hat:.6f}")
                    st.session_state["last_signal"] = {
                        "ts": ts,
                        "y_hat": y_hat,
                        "action": signal["action"],
                        "explanation": build_explanation(last_row, y_hat),
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
    last = st.session_state.get("last_signal")
    can_send = bool(last and last.get("action") in {"long", "short"} and not cfg.demo_mode)
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Подтвердить LONG", disabled=not can_send or last.get("action") != "long"):
            ts = datetime.utcnow().isoformat()
            log_action(conn, ts, "long", "user_confirmed", {})
            ok, resp = place_market_order(cfg, units=1)
            log_order_event(conn, ts, "long", 1, "sent" if ok else "skipped", resp)
            st.success(f"LONG отправлен (demo={cfg.demo_mode})")
    with action_col2:
        if st.button("Подтвердить SHORT", disabled=not can_send or last.get("action") != "short"):
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
