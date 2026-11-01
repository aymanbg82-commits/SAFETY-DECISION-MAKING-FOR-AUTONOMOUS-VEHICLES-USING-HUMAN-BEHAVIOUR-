"""
Comprehensive Evaluation Metrics for Trajectory Forecasting,
Uncertainty Quantification, and Closed-Loop Traffic Safety Simulation.
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd


# ==============================================================================
# 1. Trajectory Forecasting Metrics
# ==============================================================================

def compute_ade(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Average Displacement Error (ADE) in SI metres:
    Mean Euclidean distance across all future prediction time steps.
    y_true, y_pred: shape (N, T_pred, 2) or (T_pred, 2)
    """
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, :, :]
        y_pred = y_pred[np.newaxis, :, :]

    # Euclidean distance at each step: shape (N, T_pred)
    step_errors = np.linalg.norm(y_true - y_pred, axis=-1)
    # Mean over time steps then mean over samples
    return float(np.mean(step_errors))


def compute_fde(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Final Displacement Error (FDE) in SI metres:
    Euclidean distance at the final forecast time step T_pred.
    y_true, y_pred: shape (N, T_pred, 2) or (T_pred, 2)
    """
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, :, :]
        y_pred = y_pred[np.newaxis, :, :]

    # Euclidean distance at step -1: shape (N,)
    final_errors = np.linalg.norm(y_true[:, -1, :] - y_pred[:, -1, :], axis=-1)
    return float(np.mean(final_errors))


def evaluate_forecasting_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta_list: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Break down ADE and FDE metrics by scenario type and actor type.
    """
    assert len(y_true) == len(meta_list), "Metadata length must match sample count."

    step_errors = np.linalg.norm(y_true - y_pred, axis=-1)  # (N, T)
    sample_ades = np.mean(step_errors, axis=-1)             # (N,)
    sample_fdes = np.linalg.norm(y_true[:, -1, :] - y_pred[:, -1, :], axis=-1)  # (N,)

    rows = []
    for i, meta in enumerate(meta_list):
        rows.append({
            "scenario_type": meta.get("scenario_type", "unknown"),
            "actor_type": meta.get("actor_type", "unknown"),
            "ade": float(sample_ades[i]),
            "fde": float(sample_fdes[i])
        })

    df = pd.DataFrame(rows)
    # Aggregate by scenario and actor
    summary = df.groupby(["scenario_type", "actor_type"]).agg(
        sample_count=("ade", "count"),
        mean_ade=("ade", "mean"),
        std_ade=("ade", "std"),
        mean_fde=("fde", "mean"),
        std_fde=("fde", "std")
    ).reset_index()

    return summary


# ==============================================================================
# 2. Uncertainty Quantification Metrics
# ==============================================================================

def compute_bivariate_gaussian_nll(
    y_true: np.ndarray,
    mu: np.ndarray,
    log_sigma: np.ndarray,
    rho: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Negative Log-Likelihood (NLL) for Bivariate Gaussian trajectory predictions.
    y_true: (N, T, 2)
    mu: (N, T, 2)
    log_sigma: (N, T, 2)
    rho: (N, T, 1) or (N, T)
    """
    if rho.ndim == 3 and rho.shape[-1] == 1:
        rho = rho.squeeze(-1)

    dx = y_true[..., 0] - mu[..., 0]
    dy = y_true[..., 1] - mu[..., 1]
    sigma_x = np.exp(log_sigma[..., 0]) + eps
    sigma_y = np.exp(log_sigma[..., 1]) + eps

    # Standardized normalized residuals
    zx = dx / sigma_x
    zy = dy / sigma_y
    rho_clipped = np.clip(rho, -0.99, 0.99)
    one_minus_rho2 = 1.0 - rho_clipped**2

    # Quadratic exponent form: zx^2 - 2*rho*zx*zy + zy^2
    z = (zx**2 - 2.0 * rho_clipped * zx * zy + zy**2) / (one_minus_rho2 + eps)
    log_det = np.log(sigma_x) + np.log(sigma_y) + 0.5 * np.log(one_minus_rho2 + eps) + np.log(2.0 * np.pi)

    nll = 0.5 * z + log_det
    return float(np.mean(nll))


def compute_empirical_coverage(
    y_true: np.ndarray,
    mu: np.ndarray,
    log_sigma: np.ndarray,
    rho: np.ndarray,
    quantile: float = 0.95
) -> Tuple[float, float]:
    """
    Compute empirical coverage probability:
    Fraction of true positions falling within the elliptical confidence region of level 'quantile'.
    Also returns mean ellipse area in m^2.
    For a 2D Gaussian, Mahalanobis distance squared follows Chi-Square distribution with 2 DOF.
    Chi2(2) critical value for quantile q is: -2 * ln(1 - q).
    """
    if rho.ndim == 3 and rho.shape[-1] == 1:
        rho = rho.squeeze(-1)

    chi2_critical = -2.0 * math.log(1.0 - quantile)

    dx = y_true[..., 0] - mu[..., 0]
    dy = y_true[..., 1] - mu[..., 1]
    sigma_x = np.exp(log_sigma[..., 0])
    sigma_y = np.exp(log_sigma[..., 1])
    rho_clipped = np.clip(rho, -0.99, 0.99)
    one_minus_rho2 = 1.0 - rho_clipped**2

    zx = dx / sigma_x
    zy = dy / sigma_y
    mahalanobis_sq = (zx**2 - 2.0 * rho_clipped * zx * zy + zy**2) / (one_minus_rho2 + 1e-6)

    # Within confidence ellipse if D_M^2 <= chi2_critical
    inside_mask = (mahalanobis_sq <= chi2_critical)
    coverage = float(np.mean(inside_mask))

    # Area of ellipse: pi * sqrt(det(Sigma)) * chi2_critical = pi * sigma_x * sigma_y * sqrt(1-rho^2) * chi2_critical
    ellipse_areas = math.pi * sigma_x * sigma_y * np.sqrt(one_minus_rho2) * chi2_critical
    mean_area = float(np.mean(ellipse_areas))

    return coverage, mean_area


# ==============================================================================
# 3. Closed-Loop Simulation Safety & Efficiency Metrics
# ==============================================================================

def compute_simulation_metrics(episode_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute standard, rigorous simulation metrics across a batch of matched evaluation episodes.

    Args:
        episode_logs: List of episode outcome dictionaries with keys:
            - 'collision': bool
            - 'near_collision': bool
            - 'min_separation': float (metres)
            - 'emergency_braking_count': int
            - 'total_steps': int
            - 'task_completed': bool
            - 'elapsed_time': float (seconds)
            - 'unnecessary_intervention': bool
    """
    num_episodes = len(episode_logs)
    if num_episodes == 0:
        return {}

    collisions = sum(1 for ep in episode_logs if ep.get("collision", False))
    near_collisions = sum(1 for ep in episode_logs if ep.get("near_collision", False))
    task_completed = sum(1 for ep in episode_logs if ep.get("task_completed", False))
    unnecessary = sum(1 for ep in episode_logs if ep.get("unnecessary_intervention", False))

    min_seps = [ep["min_separation"] for ep in episode_logs if "min_separation" in ep]
    emergency_actions = [ep.get("emergency_braking_count", 0) for ep in episode_logs]
    elapsed_times = [ep.get("elapsed_time", 0.0) for ep in episode_logs]

    # Collisions per 1,000 episodes with actual denominator
    coll_rate_pct = (collisions / num_episodes) * 100.0
    coll_per_1000 = (collisions / num_episodes) * 1000.0

    return {
        "num_episodes_evaluated": num_episodes,
        "collision_count": collisions,
        "collision_rate_percent": round(coll_rate_pct, 2),
        "collisions_per_1000_episodes": round(coll_per_1000, 2),
        "near_collision_rate_percent": round((near_collisions / num_episodes) * 100.0, 2),
        "task_completion_rate_percent": round((task_completed / num_episodes) * 100.0, 2),
        "unnecessary_intervention_rate_percent": round((unnecessary / num_episodes) * 100.0, 2),
        "mean_minimum_separation_meters": round(float(np.mean(min_seps)), 3) if min_seps else 0.0,
        "std_minimum_separation_meters": round(float(np.std(min_seps)), 3) if min_seps else 0.0,
        "mean_emergency_actions_per_ep": round(float(np.mean(emergency_actions)), 2),
        "mean_elapsed_time_seconds": round(float(np.mean(elapsed_times)), 2)
    }
