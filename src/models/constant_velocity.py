"""
Constant-Velocity Trajectory Forecasting Baseline.
A transparent, deterministic physics-based forecasting baseline
assuming actors maintain their instantaneous velocity observed at the forecast horizon origin.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class ConstantVelocityForecaster:
    """
    Constant Velocity (CV) trajectory prediction baseline.
    Predicts future displacement relative to the observation endpoint:
        dx(k) = vx_endpoint * (k * dt)
        dy(k) = vy_endpoint * (k * dt)
    """

    def __init__(self, pred_steps: int = 30, dt: float = 0.1):
        self.pred_steps = pred_steps
        self.dt = dt
        self.time_horizons = np.arange(1, pred_steps + 1, dtype=np.float32) * dt  # shape (T_pred,)

    def predict(self, X_obs: np.ndarray) -> np.ndarray:
        """
        Predict future metric displacements for a batch of observation sequences.

        Args:
            X_obs: np.ndarray of shape (N, T_obs, F)
                   F features: [dx, dy, vx, vy, speed, heading]
                   The last timestep T_obs - 1 corresponds to the observation endpoint.

        Returns:
            np.ndarray of shape (N, T_pred, 2)
                   Predicted future relative displacements [dx, dy] in SI metres.
        """
        if X_obs.ndim == 2:
            X_obs = X_obs[np.newaxis, :, :]

        N, T_obs, F = X_obs.shape

        # Extract endpoint velocity components vx (idx 2) and vy (idx 3)
        vx_endpoint = X_obs[:, -1, 2]  # shape (N,)
        vy_endpoint = X_obs[:, -1, 3]  # shape (N,)

        # Broadcast across prediction horizon: (N, 1) * (1, T_pred) -> (N, T_pred)
        dx_pred = vx_endpoint[:, np.newaxis] * self.time_horizons[np.newaxis, :]
        dy_pred = vy_endpoint[:, np.newaxis] * self.time_horizons[np.newaxis, :]

        # Stack to (N, T_pred, 2)
        Y_pred = np.stack([dx_pred, dy_pred], axis=-1)
        return Y_pred

    def predict_absolute(
        self,
        ref_x: float,
        ref_y: float,
        vx: float,
        vy: float
    ) -> np.ndarray:
        """
        Predict future absolute coordinates for a single actor given endpoint state.
        Returns: np.ndarray of shape (T_pred, 2) with absolute (x, y) coordinates.
        """
        dx = vx * self.time_horizons
        dy = vy * self.time_horizons
        x_fut = ref_x + dx
        y_fut = ref_y + dy
        return np.column_stack([x_fut, y_fut])
