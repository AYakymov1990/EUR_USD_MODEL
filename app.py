import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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


def fetch_news_articles(cfg: CRMConfig, limit: int = 5) -> list:
    api_key = cfg.news_api_key or os.environ.get("NEWS_API_KEY")
    if not api_key:
        return []
    # Top-headlines w NewsAPI wymaga country lub sources; używamy country=us dla poprawnego zapytania
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

    # Fallback: everything (gdy top-headlines nic nie zwrócił)
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


def send_signal_email(cfg: CRMConfig, signal: dict, meta: dict, y_hat: float, ts: str) -> None:
    if not cfg.allow_email:
        return
    direction = signal.get("action")
    if direction not in {"long", "short"}:
        return
    subject = f"Trader CRM: sygnał {direction.upper()} @ {ts}"
    body = (
        f"Czas: {ts}\n"
        f"Działanie: {direction}\n"
        f"Prognoza y_hat: {y_hat}\n"
        f"Powód: {signal.get('reason','')}\n"
        f"Tryb: {signal.get('regime','')}\n"
        f"Cena: {meta.get('close')}"
    )
    ok, err = send_email(cfg, subject, body)
    if not ok:
        st.warning(f"Nie udało się wysłać e-maila: {err}")


def main() -> None:
    cfg = get_cfg()
    sel = get_selected(cfg)
    conn = get_db(cfg)

    st.set_page_config(page_title="Trader CRM", layout="wide")
    st.title("Trader CRM")
    mode_label = "LIVE (practice)" if not cfg.demo_mode else "DEMO"
    st.caption(f"Tryb: {mode_label}. Live wymaga .env z kluczami OANDA.")

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
            st.sidebar.error(f"Nie udało się załadować artefaktów: {exc}")
    else:
        st.sidebar.warning("Model/skaler nie znaleziony. Sygnały będą fikcyjne.")

    qp = _get_query_params()
    auto_default = qp.get("auto", ["0"])[0] in {"1", "true", "yes"}
    if "auto_mode" not in st.session_state:
        st.session_state["auto_mode"] = auto_default
    auto_mode = st.session_state.get("auto_mode", False)

    news_articles = fetch_news_articles(cfg)
    acct = fetch_account(cfg)

    tabs = st.tabs(["Dashboard", "Wiadomości", "Dziennik sygnałów", "Statystyka", "Ustawienia"])
    # Wiadomości
    with tabs[1]:
        st.subheader("Wiadomości EUR/USD")
        if news_articles:
            for art in news_articles:
                dt = art.get("publishedAt", "")[:10]
                title = art.get("title", "No title")
                url = art.get("url", "#")
                st.markdown(f"- {dt} — [{title}]({url})")
        else:
            st.caption("Wiadomości niedostępne (brak NEWS_API_KEY lub błąd zapytania).")

    # Ustawienia
    with tabs[4]:
        st.subheader("Ustawienia")
        st.write(f"Tryb: {'LIVE' if not cfg.demo_mode else 'DEMO'}; strategia: {cfg.signal_mode.upper()}")
        auto_mode = st.checkbox("Tryb auto (co 15m)", value=auto_mode)
        if auto_mode != st.session_state.get("auto_mode"):
            st.session_state["auto_mode"] = auto_mode
            qp["auto"] = "1" if auto_mode else "0"
            _set_query_params(qp)
        st.caption("Aby zmienić demo/live, zaktualizuj .env i uruchom ponownie.")
        if cfg.signal_mode != "fib":
            thr_default = sel.get("threshold", 0.0) if sel else 0.0
            thr_override = st.slider(
                "Próg sygnału (threshold)", min_value=-0.001, max_value=0.001, step=0.00001, value=float(st.session_state.get("threshold_override", thr_default))
            )
            st.session_state["threshold_override"] = thr_override
        else:
            st.caption("Strategia fib nie używa progów.")

    auto_mode = st.session_state.get("auto_mode", False)

    # Dashboard
    with tabs[0]:
        st.subheader("Konto (EUR/USD)")
        st.json(acct)

        st.subheader("Sygnały")
        st_autorefresh = getattr(st, "autorefresh", None)
        if st_autorefresh and auto_mode:
            st_autorefresh(interval=60_000, key="auto_refresh")
        elif auto_mode:
            st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

        now = datetime.utcnow()
        last_run = st.session_state.get("last_auto_run")
        ready_for_auto = not last_run or (now - last_run) > timedelta(minutes=1)

        manual_clicked = st.button("Pobierz sygnał")
        should_run = manual_clicked or (auto_mode and ready_for_auto)
        if auto_mode and should_run:
            st.session_state["last_auto_run"] = now

        if cfg.demo_mode:
            if should_run:
                rows = st.session_state["demo_rows"]
                if not rows:
                    st.error("Brak danych demo.")
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
                    sel_current = dict(sel)
                    if "threshold_override" in st.session_state:
                        sel_current["threshold"] = st.session_state["threshold_override"]
                    signal = make_signal(row, y_hat, sel_current)
                    conf = confidence_from_pred(y_hat, row.get("pred_abs_p95"))
                    ts = row.get("time", datetime.utcnow()).isoformat()
                    log_signal(conn, ts, y_hat, signal["action"], conf, signal)
                    st.success(f"Sygnał {signal['action']} @ {y_hat:.6f}")
                    st.session_state["last_signal"] = {
                        "ts": ts,
                        "y_hat": y_hat,
                        "action": signal["action"],
                        "explanation": build_explanation(row, y_hat),
                        "confidence": conf,
                    }
        else:
            if should_run:
                df_m15, df_h1 = get_latest_live_window(cfg)
                if df_m15 is None or df_h1 is None or df_m15.empty or df_h1.empty:
                    st.error("Nie udało się pobrać świec OANDA. Sprawdź sieć/klucze.")
                else:
                    try:
                        feature_df, last_row, meta = build_live_feature_row(df_m15, df_h1)
                    except Exception as exc:
                        st.error(f"Błąd przygotowania cech: {exc}")
                    else:
                        if artifacts_ok:
                            pred_info = run_inference(model, scaler, feature_df.values)
                            y_hat = pred_info["y_hat_scalar"]
                        else:
                            y_hat = float(last_row.get("target", 0.0))
                        sel_current = dict(sel)
                        if "threshold_override" in st.session_state:
                            sel_current["threshold"] = st.session_state["threshold_override"]
                        signal = make_signal(last_row, y_hat, sel_current)
                        conf = confidence_from_pred(y_hat)
                        ts_val = meta.get("time") or datetime.utcnow()
                        ts = ts_val.isoformat()
                        last_logged = get_last_signal_time(conn)
                        if last_logged and last_logged == ts:
                            if manual_clicked:
                                st.info("Sygnał dla tego bara już zapisany.")
                        else:
                            log_signal(conn, ts, y_hat, signal["action"], conf, {**signal, **meta})
                            send_signal_email(cfg, signal, meta, y_hat, ts)
                            st.success(f"Sygnał {signal['action']} @ {y_hat:.6f}")
                        st.session_state["last_signal"] = {
                            "ts": ts,
                            "y_hat": y_hat,
                            "action": signal["action"],
                            "explanation": build_explanation(last_row, y_hat),
                            "confidence": conf,
                        }

        st.subheader("Ostatni sygnał")
        last = st.session_state.get("last_signal")
        if last:
            st.write(last["ts"])
            st.metric("Działanie", last["action"], f"{last['y_hat']:.6f}")
            st.progress(min(max(last["confidence"], 0), 1.0))
            st.text(last["explanation"])
        else:
            st.info("Brak sygnałów.")

        last = st.session_state.get("last_signal")
        can_send = bool(last and last.get("action") in {"long", "short"} and not cfg.demo_mode)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Potwierdź LONG", disabled=not can_send or last.get("action") != "long"):
                ts = datetime.utcnow().isoformat()
                log_action(conn, ts, "long", "user_confirmed", {})
                ok, resp = place_market_order(cfg, units=1)
                log_order_event(conn, ts, "long", 1, "sent" if ok else "skipped", resp)
                st.success(f"LONG wysłany (demo={cfg.demo_mode})")
        with col2:
            if st.button("Potwierdź SHORT", disabled=not can_send or last.get("action") != "short"):
                ts = datetime.utcnow().isoformat()
                log_action(conn, ts, "short", "user_confirmed", {})
                ok, resp = place_market_order(cfg, units=-1)
                log_order_event(conn, ts, "short", -1, "sent" if ok else "skipped", resp)
                st.success(f"SHORT wysłany (demo={cfg.demo_mode})")

    # Dziennik
    with tabs[2]:
        st.subheader("Dziennik sygnałów")
        signals = fetch_recent_signals(conn, limit=100)
        if signals:
            df_sig = pd.DataFrame(signals)
            st.dataframe(df_sig)
        else:
            st.info("Dziennik pusty.")

    # Statystyka
    with tabs[3]:
        st.subheader("Statystyka")
        signals = fetch_recent_signals(conn, limit=500)
        if signals:
            df_sig = pd.DataFrame(signals)
            st.write(f"Łącznie sygnałów: {len(df_sig)}")
            if "action" in df_sig.columns:
                st.write(df_sig["action"].value_counts())
            else:
                st.info("Kolumna action nie występuje w danych sygnałów.")
        else:
            st.info("Brak danych do statystyk.")


if __name__ == "__main__":
    main()
