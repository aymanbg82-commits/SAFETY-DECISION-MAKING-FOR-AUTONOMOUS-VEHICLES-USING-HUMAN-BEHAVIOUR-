"""
Unit tests for forecasting, uncertainty, and simulation metrics.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_ade,
    compute_fde,
    compute_bivariate_gaussian_nll,
    compute_empirical_coverage,
    compute_simulation_metrics
)


def test_ade_and_fde_known_displacements():
    # True: (0, 0) at all 10 steps
    # Pred: (3, 4) at all 10 steps (Euclidean dist = 5.0m)
    y_true = np.zeros((2, 10, 2), dtype=np.float32)
    y_pred = np.full((2, 10, 2), [3.0, 4.0], dtype=np.float32)

    ade = compute_ade(y_true, y_pred)
    fde = compute_fde(y_true, y_pred)

    assert ade == pytest.approx(5.0, abs=1e-5)
    assert fde == pytest.approx(5.0, abs=1e-5)


def test_bivariate_gaussian_coverage():
    # If predictions equal true with zero covariance scale, coverage should be 1.0
    y_true = np.zeros((100, 10, 2), dtype=np.float32)
    mu = np.zeros((100, 10, 2), dtype=np.float32)
    log_sigma = np.zeros((100, 10, 2), dtype=np.float32)  # sigma = 1.0
    rho = np.zeros((100, 10), dtype=np.float32)

    coverage, area = compute_empirical_coverage(y_true, mu, log_sigma, rho, quantile=0.95)
    assert coverage == 1.0
    assert area > 0.0


def test_simulation_metrics_calculation():
    logs = [
        {"collision": False, "near_collision": False, "task_completed": True, "min_separation": 4.5, "emergency_braking_count": 0, "elapsed_time": 5.0, "unnecessary_intervention": False},
        {"collision": True, "near_collision": True, "task_completed": False, "min_separation": 0.0, "emergency_braking_count": 2, "elapsed_time": 3.2, "unnecessary_intervention": False},
        {"collision": False, "near_collision": True, "task_completed": True, "min_separation": 1.2, "emergency_braking_count": 1, "elapsed_time": 5.0, "unnecessary_intervention": True},
        {"collision": False, "near_collision": False, "task_completed": True, "min_separation": 3.8, "emergency_braking_count": 0, "elapsed_time": 5.0, "unnecessary_intervention": False},
    ]

    res = compute_simulation_metrics(logs)
    assert res["num_episodes_evaluated"] == 4
    assert res["collision_count"] == 1
    assert res["collision_rate_percent"] == 25.0
    assert res["collisions_per_1000_episodes"] == 250.0
    assert res["task_completion_rate_percent"] == 75.0
    assert res["near_collision_rate_percent"] == 50.0
    assert res["unnecessary_intervention_rate_percent"] == 25.0
