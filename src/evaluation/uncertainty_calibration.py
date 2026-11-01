"""Validation-only calibration for constant-velocity forecast residuals."""

from typing import Any, Dict
import math
import numpy as np


def calibrate_cv_residual_envelope(
    x_obs: np.ndarray,
    y_future: np.ndarray,
    quantile: float = 0.90,
    dt: float = 0.1,
) -> Dict[str, Any]:
    """Calibrate per-horizon isotropic radii from held-out CV residuals.

    The returned radius is a marginal empirical quantile of Euclidean forecast
    error at each horizon. It is not a joint trajectory guarantee and assumes
    future residuals are exchangeable with the validation residuals.
    """
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0")
    if x_obs.ndim != 3 or y_future.ndim != 3 or x_obs.shape[0] != y_future.shape[0]:
        raise ValueError("x_obs and y_future must be aligned three-dimensional arrays")

    vx = x_obs[:, -1, 2:3]
    vy = x_obs[:, -1, 3:4]
    horizons = np.arange(1, y_future.shape[1] + 1, dtype=float) * dt
    cv = np.stack([vx * horizons, vy * horizons], axis=-1)
    residual_radius = np.linalg.norm(y_future - cv, axis=-1)
    radii = np.quantile(residual_radius, quantile, axis=0)
    chi2_radius = math.sqrt(-2.0 * math.log(1.0 - quantile))
    sigma = np.maximum(radii / chi2_radius, 1e-3)
    return {
        "method": "cv_residual_euclidean_quantile",
        "quantile": float(quantile),
        "coverage_semantics": "marginal per-horizon empirical coverage; not joint trajectory coverage",
        "assumption": "validation residuals are exchangeable with future residuals",
        "dt": float(dt),
        "horizons": [round(float(h), 3) for h in horizons],
        "sigma_x": [float(value) for value in sigma],
        "sigma_y": [float(value) for value in sigma],
        "rho": [0.0] * len(sigma),
        "validation_sample_count": int(x_obs.shape[0]),
    }


def evaluate_cv_residual_calibration(
    x_obs: np.ndarray,
    y_future: np.ndarray,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure empirical per-horizon coverage and mean ellipse area."""
    vx = x_obs[:, -1, 2:3]
    vy = x_obs[:, -1, 3:4]
    horizons = np.asarray(profile["horizons"], dtype=float)
    cv = np.stack([vx * horizons, vy * horizons], axis=-1)
    residual_radius = np.linalg.norm(y_future - cv, axis=-1)
    quantile = float(profile["quantile"])
    radii = np.asarray(profile["sigma_x"], dtype=float) * math.sqrt(-2.0 * math.log(1.0 - quantile))
    coverage_by_horizon = np.mean(residual_radius <= radii[None, :], axis=0)
    area_by_horizon = math.pi * np.asarray(profile["sigma_x"]) * np.asarray(profile["sigma_y"]) * (-2.0 * math.log(1.0 - quantile))
    return {
        "quantile": quantile,
        "mean_marginal_coverage_percent": float(np.mean(coverage_by_horizon) * 100.0),
        "minimum_marginal_coverage_percent": float(np.min(coverage_by_horizon) * 100.0),
        "final_horizon_coverage_percent": float(coverage_by_horizon[-1] * 100.0),
        "mean_ellipse_area_m2": float(np.mean(area_by_horizon)),
        "coverage_by_horizon_percent": [float(value * 100.0) for value in coverage_by_horizon],
    }