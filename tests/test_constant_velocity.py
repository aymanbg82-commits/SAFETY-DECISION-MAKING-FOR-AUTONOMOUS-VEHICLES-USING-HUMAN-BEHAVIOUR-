"""
Unit tests for Constant-Velocity Forecaster.
"""

import pytest
import numpy as np
from src.models.constant_velocity import ConstantVelocityForecaster
from src.evaluation.metrics import compute_ade, compute_fde


def test_constant_velocity_linear_prediction():
    # 1 actor moving at vx=10 m/s, vy=0 m/s for 20 observation steps
    # dt = 0.1s, pred_steps = 30
    cv = ConstantVelocityForecaster(pred_steps=30, dt=0.1)

    # Observation: 20 steps, vx=10, vy=2
    X_obs = np.zeros((1, 20, 6), dtype=np.float32)
    X_obs[0, :, 2] = 10.0  # vx
    X_obs[0, :, 3] = 2.0   # vy

    Y_pred = cv.predict(X_obs)  # shape (1, 30, 2)

    assert Y_pred.shape == (1, 30, 2)

    # At step 10 (t = 1.0s), dx should be 10.0m, dy should be 2.0m
    np.testing.assert_allclose(Y_pred[0, 9, 0], 10.0, atol=1e-5)
    np.testing.assert_allclose(Y_pred[0, 9, 1], 2.0, atol=1e-5)

    # At step 30 (t = 3.0s), dx should be 30.0m, dy should be 6.0m
    np.testing.assert_allclose(Y_pred[0, 29, 0], 30.0, atol=1e-5)
    np.testing.assert_allclose(Y_pred[0, 29, 1], 6.0, atol=1e-5)


def test_cv_zero_error_on_linear_ground_truth():
    cv = ConstantVelocityForecaster(pred_steps=30, dt=0.1)
    # Ground truth moving with constant velocity
    time_horizons = np.arange(1, 31) * 0.1
    Y_true = np.zeros((1, 30, 2), dtype=np.float32)
    Y_true[0, :, 0] = 12.0 * time_horizons
    Y_true[0, :, 1] = -3.0 * time_horizons

    X_obs = np.zeros((1, 20, 6), dtype=np.float32)
    X_obs[0, -1, 2] = 12.0
    X_obs[0, -1, 3] = -3.0

    Y_pred = cv.predict(X_obs)

    ade = compute_ade(Y_true, Y_pred)
    fde = compute_fde(Y_true, Y_pred)

    assert ade == pytest.approx(0.0, abs=1e-5)
    assert fde == pytest.approx(0.0, abs=1e-5)
