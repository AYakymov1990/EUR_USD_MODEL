"""PyTorch regression models and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
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


class MLPRegressor(nn.Module):
    """Simple MLP regressor for tabular features."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Iterable[int] = (64, 32),
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainConfig:
    """Training configuration for regression."""

    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
    seed: int = 42
    device: str = "cpu"
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 0.0
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.3
    loss_name: str = "mse"
    huber_delta: float = 1.0


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


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-cosh loss."""
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff)))


def _get_loss_fn(loss_name: str, huber_delta: float) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=huber_delta)
    if loss_name == "logcosh":
        return None
    raise ValueError(f"Unknown loss: {loss_name}")


def compute_regression_metrics(pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute basic regression metrics."""
    mse = float(np.mean((pred - y_true) ** 2))
    mae = float(np.mean(np.abs(pred - y_true)))
    corr = float(np.corrcoef(pred, y_true)[0, 1])
    dir_acc = float(np.mean(np.sign(pred) == np.sign(y_true)))
    return {"mse": mse, "mae": mae, "corr": corr, "dir_acc": dir_acc}


def _train(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> Tuple[nn.Module, Dict[str, list[float]]]:
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

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = _get_loss_fn(config.loss_name, config.huber_delta)

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
            if config.loss_name == "logcosh":
                loss = log_cosh_loss(preds, batch_y)
            else:
                loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            preds_val = model(x_val_t.to(device))
            if config.loss_name == "logcosh":
                val_loss = log_cosh_loss(preds_val, y_val_t.to(device)).item()
            else:
                val_loss = loss_fn(preds_val, y_val_t.to(device)).item()

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


def train_linear_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig | None = None,
) -> Tuple[LinearRegressionModel, Dict[str, list[float]]]:
    """Train a linear regression model with MSE loss and early stopping."""
    if config is None:
        config = TrainConfig(loss_name="mse", dropout_rate=0.0, hidden_sizes=())
    model = LinearRegressionModel(n_features=x_train.shape[1])
    return _train(model, x_train, y_train, x_val, y_val, config)


def train_mlp_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> Tuple[MLPRegressor, Dict[str, list[float]]]:
    """Train an MLP regressor with early stopping."""
    model = MLPRegressor(
        n_features=x_train.shape[1],
        hidden_sizes=config.hidden_sizes,
        dropout_rate=config.dropout_rate,
    )
    return _train(model, x_train, y_train, x_val, y_val, config)


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Predict using a trained model."""
    model.eval()
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        preds = model(x_t).cpu().numpy()
    return preds


def train_and_select_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split_cfg: dict[str, float],
    search_cfg: dict[str, Iterable],
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train and select the best model by validation MSE."""
    x = df[feature_cols].values
    y = df[target_col].values

    n = len(df)
    train_end = int(n * split_cfg.get("train", 0.7))
    val_end = int(n * (split_cfg.get("train", 0.7) + split_cfg.get("val", 0.15)))

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    best_model: nn.Module | None = None
    best_metrics: Dict[str, float] | None = None
    best_params: dict[str, Any] | None = None
    best_val_mse = float("inf")
    history_map: dict[str, Dict[str, list[float]]] = {}

    for params in search_cfg.get("param_grid", []):
        config = TrainConfig(**params)
        model, history = train_mlp_model(x_train_s, y_train, x_val_s, y_val, config)

        pred_val = predict(model, x_val_s)
        pred_test = predict(model, x_test_s)

        metrics_val = compute_regression_metrics(pred_val, y_val)
        metrics_test = compute_regression_metrics(pred_test, y_test)

        history_map[str(params)] = history
        if metrics_val["mse"] < best_val_mse:
            best_val_mse = metrics_val["mse"]
            best_model = model
            best_metrics = {
                "val": metrics_val,
                "test": metrics_test,
            }
            best_params = params

    if best_model is None or best_metrics is None or best_params is None:
        raise ValueError("No models trained. Provide a non-empty param_grid.")

    artifacts = {
        "best_params": best_params,
        "scaler": {"mean": scaler.mean_.tolist(), "std": scaler.std_.tolist()},
        "metrics": best_metrics,
        "history": history_map,
    }
    return best_model, artifacts
