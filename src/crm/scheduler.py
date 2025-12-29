"""Simple scheduler loop for demo/live polling."""

from __future__ import annotations

import threading
import time
from typing import Callable


def run_loop(interval_sec: float, fn: Callable[[], None], stop_after: float | None = None) -> None:
    """Run fn every interval_sec seconds; stop_after in seconds (optional)."""
    start = time.time()
    while True:
        fn()
        if stop_after is not None and (time.time() - start) > stop_after:
            break
        time.sleep(interval_sec)


def run_in_thread(interval_sec: float, fn: Callable[[], None]) -> threading.Thread:
    """Run loop in background thread (no stop condition)."""
    t = threading.Thread(target=run_loop, args=(interval_sec, fn), daemon=True)
    t.start()
    return t
