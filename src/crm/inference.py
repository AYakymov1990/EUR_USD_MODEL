"""Inference helpers for loading artifacts and producing predictions."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.serialization import add_safe_globals

from src.model import predict, MLPRegressor


def load_artifacts(model_path: Path, scaler_path: Path) -> Tuple[Any, Any]:
    """Load model (torch) and scaler (pickle)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
    # Allowlist our model class for safe loading
    add_safe_globals([MLPRegressor])
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        # For older torch without weights_only param
        model = torch.load(model_path, map_location="cpu")
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    model.eval()
    return model, scaler


def run_inference(model: Any, scaler: Any, features: np.ndarray) -> Dict[str, Any]:
    """Scale features and run model prediction."""
    if features.ndim == 1:
        features = features.reshape(1, -1)
    feats_s = scaler.transform(features)
    y_hat = predict(model, feats_s)
    return {
        "y_hat": y_hat,
        "y_hat_scalar": float(np.mean(y_hat)),
        "n": int(len(y_hat)),
    }
