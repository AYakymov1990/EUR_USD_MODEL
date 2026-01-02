"""Configuration loader for Trader CRM."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


@dataclass
class CRMConfig:
    """Runtime configuration for CRM/demo."""

    demo_mode: bool = True
    oanda_env: str = "practice"
    oanda_api_key: Optional[str] = None
    oanda_account_id: Optional[str] = None
    instrument: str = "EUR_USD"
    granularity: str = "M15"
    timeout: int = 15
    artifacts_dir: Path = Path("data/artifacts")
    features_path: Path = Path("data/eurusd_features.parquet")
    selected_config_path: Path = Path("data/artifacts/selected_config.json")
    sqlite_path: Path = Path("data/artifacts/trader_crm.sqlite")
    max_demo_rows: int = 2_000
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_user: Optional[str] = None
    email_password: Optional[str] = None
    email_to: Optional[str] = None
    email_from: Optional[str] = None

    @property
    def allow_email(self) -> bool:
        return all([self.email_smtp_host, self.email_user, self.email_password, self.email_to])

    @property
    def allow_live(self) -> bool:
        return not self.demo_mode and bool(self.oanda_api_key) and bool(self.oanda_account_id)


def load_config(env: Optional[Dict[str, Any]] = None, config_path: Optional[Path] = None) -> CRMConfig:
    """Load configuration from env (and optional json file)."""
    _load_dotenv_if_available()
    env = env or os.environ
    cfg_json: Dict[str, Any] = {}
    if config_path and config_path.exists():
        try:
            cfg_json = json.loads(config_path.read_text())
        except Exception:
            cfg_json = {}

    def _get(key: str, default: Any) -> Any:
        return env.get(key, cfg_json.get(key, default))

    demo_mode = str(_get("DEMO_MODE", "true")).lower() in {"1", "true", "yes"}
    return CRMConfig(
        demo_mode=demo_mode,
        oanda_env=str(_get("OANDA_ENV", "practice")),
        oanda_api_key=_get("OANDA_API_KEY", None),
        oanda_account_id=_get("OANDA_ACCOUNT_ID", None),
        instrument=str(_get("OANDA_INSTRUMENT", "EUR_USD")),
        granularity=str(_get("OANDA_GRANULARITY", "M15")),
        timeout=int(_get("OANDA_TIMEOUT", 15)),
        artifacts_dir=Path(_get("ARTIFACTS_DIR", "data/artifacts")),
        features_path=Path(_get("FEATURES_PATH", "data/eurusd_features.parquet")),
        selected_config_path=Path(_get("SELECTED_CONFIG", "data/artifacts/selected_config.json")),
        sqlite_path=Path(_get("SQLITE_PATH", "data/artifacts/trader_crm.sqlite")),
        max_demo_rows=int(_get("MAX_DEMO_ROWS", 2_000)),
        email_smtp_host=_get("EMAIL_SMTP_HOST", None),
        email_smtp_port=int(_get("EMAIL_SMTP_PORT", 587)),
        email_user=_get("EMAIL_USER", None),
        email_password=_get("EMAIL_PASSWORD", None),
        email_to=_get("EMAIL_TO", None),
        email_from=_get("EMAIL_FROM", None),
    )
