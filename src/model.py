"""PyTorch linear regression model and training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """Simple linear regression model."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class TrainConfig:
    """Training configuration for linear regression."""

    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64


def train_linear_model(
    x: np.ndarray, y: np.ndarray, config: TrainConfig | None = None
) -> Tuple[LinearRegressionModel, list[float]]:
    """Train a linear regression model with MSE loss.

    Args:
        x: Feature matrix (num_samples, num_features).
        y: Target vector (num_samples,).
        config: Training configuration.

    Returns:
        Trained model and list of epoch losses.
    """
    if config is None:
        config = TrainConfig()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model = LinearRegressionModel(n_features=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        losses.append(epoch_loss / len(dataset))

    return model, losses
