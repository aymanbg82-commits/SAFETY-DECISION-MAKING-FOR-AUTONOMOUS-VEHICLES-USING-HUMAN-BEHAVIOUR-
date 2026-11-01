"""
Uncertainty-Aware Trajectory Forecaster (Bivariate Gaussian GRU).
Outputs predictive mean displacements (mu_x, mu_y), log standard deviations (log_sigma_x, log_sigma_y),
and cross-correlation coefficients (rho) trained using Negative Log-Likelihood (NLL).
"""

import os
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


class BivariateGaussianLoss(nn.Module):
    """
    Negative Log-Likelihood (NLL) Loss for Bivariate Gaussian trajectory distributions.
    Loss = 0.5 * (zx^2 - 2*rho*zx*zy + zy^2) / (1 - rho^2) + log(sigma_x) + log(sigma_y) + 0.5*log(1 - rho^2) + log(2*pi)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        rho: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        dx = y_true[..., 0] - mu[..., 0]
        dy = y_true[..., 1] - mu[..., 1]

        sigma_x = torch.exp(log_sigma[..., 0]) + self.eps
        sigma_y = torch.exp(log_sigma[..., 1]) + self.eps

        zx = dx / sigma_x
        zy = dy / sigma_y

        rho_clipped = torch.clamp(rho, -0.99, 0.99)
        one_minus_rho2 = 1.0 - rho_clipped**2 + self.eps

        z = (zx**2 - 2.0 * rho_clipped * zx * zy + zy**2) / one_minus_rho2
        log_det = (
            log_sigma[..., 0]
            + log_sigma[..., 1]
            + 0.5 * torch.log(one_minus_rho2)
            + math.log(2.0 * math.pi)
        )

        nll = 0.5 * z + log_det
        return torch.mean(nll)


class UncertaintyTrajectoryGRU(nn.Module):
    """
    Uncertainty-Aware GRU Forecaster.
    Encodes observation sequence and decodes 5 parameters per prediction step:
    - mu_x, mu_y: predicted mean metric displacement
    - log_sigma_x, log_sigma_y: log scale uncertainty
    - logits_rho: unconstrained correlation logit -> rho in (-1, 1) via tanh
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        pred_steps: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_steps = pred_steps

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 5 outputs per prediction step: [mu_x, mu_y, log_sigma_x, log_sigma_y, logits_rho]
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_steps * 5)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        out, h_n = self.encoder(x)
        last_hidden = h_n[-1]

        raw_params = self.fc_head(last_hidden).view(batch_size, self.pred_steps, 5)

        mu = raw_params[..., 0:2]
        # Clamp log_sigma to prevent numerical explosion or collapse
        log_sigma = torch.clamp(raw_params[..., 2:4], min=-3.0, max=3.0)
        rho = torch.tanh(raw_params[..., 4])

        return mu, log_sigma, rho


class UncertaintyForecasterTrainer:
    """
    Trainer and evaluation coordinator for UncertaintyTrajectoryGRU.
    """

    def __init__(
        self,
        model: Optional[UncertaintyTrajectoryGRU] = None,
        device: Optional[str] = None,
        lr: float = 0.001
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model or UncertaintyTrajectoryGRU()).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = BivariateGaussianLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for x_b, y_b in dataloader:
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            self.optimizer.zero_grad()
            mu, log_sigma, rho = self.model(x_b)
            loss = self.criterion(mu, log_sigma, rho, y_b)
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
                mu, log_sigma, rho = self.model(x_b)
                loss = self.criterion(mu, log_sigma, rho, y_b)
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

        history = {"train_nll": [], "val_nll": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss = self.evaluate(val_loader)

            history["train_nll"].append(t_loss)
            history["val_nll"].append(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        return history

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic trajectory forecasts.
        Returns:
            Dict with 'mu', 'log_sigma', 'sigma', and 'rho' arrays.
        """
        self.model.eval()
        if X.ndim == 2:
            X = X[np.newaxis, :, :]

        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            mu, log_sigma, rho = self.model(x_t)

            mu_np = mu.cpu().numpy()
            log_sig_np = log_sigma.cpu().numpy()
            rho_np = rho.cpu().numpy()
            sig_np = np.exp(log_sig_np)

            return {
                "mu": mu_np,
                "log_sigma": log_sig_np,
                "sigma": sig_np,
                "rho": rho_np
            }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Uncertainty model saved to {path}")

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Uncertainty model loaded from {path}")
