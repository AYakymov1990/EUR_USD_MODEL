"""Utilities for loading and saving project datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DATA_DIR = Path("data")


def _ensure_parent(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, relative_path: str) -> Path:
    """Save a DataFrame to CSV or Parquet under the data/ directory.

    Args:
        df: DataFrame to save.
        relative_path: Path relative to data/, e.g. "raw/eurusd_m15.csv".

    Returns:
        Full path to the saved file.
    """
    path = DATA_DIR / relative_path
    _ensure_parent(path)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported file extension. Use .csv or .parquet")

    return path


def load_df(relative_path: str, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load a DataFrame from CSV or Parquet under the data/ directory.

    Args:
        relative_path: Path relative to data/, e.g. "raw/eurusd_m15.csv".
        parse_dates: Optional list of columns to parse as dates for CSV.

    Returns:
        Loaded DataFrame.
    """
    path = DATA_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=parse_dates)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError("Unsupported file extension. Use .csv or .parquet")
