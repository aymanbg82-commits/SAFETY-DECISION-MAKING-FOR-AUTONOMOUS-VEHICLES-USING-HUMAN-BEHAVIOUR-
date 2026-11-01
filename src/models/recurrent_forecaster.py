"""
Compact Recurrent Trajectory Forecaster (PyTorch GRU).
Maps observation sequences (T_obs = 20, F = 6) to future displacements (T_pred = 30, D = 2).
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


class TrajectoryGRU(nn.Module):
    """
    Compact GRU Sequence-to-Sequence Forecaster.
    Encodes 2.0s observation history and decodes 3.0s future metric trajectory.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        pred_steps: int = 30,
        output_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_steps = pred_steps
        self.output_dim = output_dim

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Multi-step projection head: hidden_dim -> (pred_steps * output_dim)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_steps * output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: torch.Tensor of shape (batch_size, T_obs, input_dim)
        Returns:
            torch.Tensor of shape (batch_size, T_pred, output_dim)
        """
        batch_size = x.size(0)
        out, h_n = self.encoder(x)
        # Use last layer hidden state: h_n[-1] of shape (batch_size, hidden_dim)
        last_hidden = h_n[-1]
        flat_pred = self.fc_head(last_hidden)  # (batch_size, pred_steps * 2)
        return flat_pred.view(batch_size, self.pred_steps, self.output_dim)


class RecurrentForecasterTrainer:
    """
    Trainer and inference manager for TrajectoryGRU.
    """

    def __init__(
        self,
        model: Optional[TrajectoryGRU] = None,
        device: Optional[str] = None,
        lr: float = 0.001
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model or TrajectoryGRU()).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for x_b, y_b in dataloader:
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x_b)
            loss = self.criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item() * len(x_b)
            count += len(x_b)

        return total_loss / max(1, count)

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for x_b, y_b in dataloader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                pred = self.model(x_b)
                loss = self.criterion(pred, y_b)
                total_loss += loss.item() * len(x_b)
                count += len(x_b)

        return total_loss / max(1, count)

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        epochs: int = 25,
        batch_size: int = 64,
        patience: int = 5,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss = self.evaluate(val_loader)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        if X.ndim == 2:
            X = X[np.newaxis, :, :]

        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(x_t)
            return pred.cpu().numpy()

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {path}")
