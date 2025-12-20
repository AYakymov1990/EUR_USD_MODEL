"""PyTorch linear regression model and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class StandardScaler:
    """Lightweight standard scaler (fit on train only)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """Fit mean and std on the given array."""
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std = np.where(std == 0, 1.0, std)
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform using the fitted mean/std."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler is not fitted.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(x)
        return self.transform(x)


class LinearRegressionModel(nn.Module):
    """Simple linear regression model."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


@dataclass
class TrainConfig:
    """Training configuration for linear regression."""

    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
    seed: int = 42
    device: str = "cpu"
    weight_decay: float = 1e-4
    patience: int = 15
    min_delta: float = 0.0


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse_baseline_zero(y: np.ndarray) -> float:
    """Baseline MSE for always predicting zero."""
    return float(np.mean(y**2))


def mse_baseline_mean(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Baseline MSE for predicting train mean."""
    mean_val = float(np.mean(y_train))
    return float(np.mean((y_test - mean_val) ** 2))


def train_linear_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig | None = None,
) -> Tuple[LinearRegressionModel, Dict[str, list[float]]]:
    """Train a linear regression model with MSE loss and early stopping."""
    if config is None:
        config = TrainConfig()

    _set_seeds(config.seed)
    device = torch.device(config.device)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(x_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=False
    )

    model = LinearRegressionModel(n_features=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = config.patience

    for _ in range(config.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            preds_val = model(x_val_t.to(device))
            val_loss = criterion(preds_val, y_val_t.to(device)).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss + config.min_delta < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict(model: LinearRegressionModel, x: np.ndarray) -> np.ndarray:
    """Predict using a trained model."""
    model.eval()
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        preds = model(x_t).cpu().numpy()
    return preds
