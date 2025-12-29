"""CRM layer for Trader demo (signals, explanations, execution)."""

from .config import CRMConfig, load_config  # noqa: F401
from .data_feed import demo_feed  # noqa: F401
from .features_adapter import build_feature_row  # noqa: F401
from .inference import load_artifacts, run_inference  # noqa: F401
from .signals import load_selected_config, make_signal  # noqa: F401
from .storage import (
    ensure_schema,
    get_connection,
    log_action,
    log_order_event,
    log_signal,
    fetch_recent_signals,
)  # noqa: F401
