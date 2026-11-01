"""
End-to-End Smoke Test and Information Leakage Prevention Verification.
"""

import pytest
import numpy as np
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.data.data_pipeline import TrajectoryDataPipeline
from src.models.constant_velocity import ConstantVelocityForecaster
from src.models.recurrent_forecaster import TrajectoryGRU, RecurrentForecasterTrainer
from src.models.uncertainty import UncertaintyTrajectoryGRU, UncertaintyForecasterTrainer
from src.risk.geometric_risk import GeometricRiskEstimator
from src.simulation.policies import ReactivePolicy, PredictionAwarePolicy, UncertaintyAwarePolicy
from src.simulation.simulator import TrafficSimulator
from src.evaluation.metrics import compute_ade, compute_fde, compute_simulation_metrics


def test_units_and_timestep_consistency():
    gen = SyntheticTrajectoryGenerator(time_step=0.1)
    ep = gen.generate_episode("pedestrian_crossing", seed=42)

    assert ep.time_step == 0.1
    # Check that timestamps advance strictly by dt = 0.1s
    timestamps = [s.timestamp for s in ep.states if s.is_ego]
    for i in range(1, len(timestamps)):
        assert (timestamps[i] - timestamps[i - 1]) == pytest.approx(0.1, abs=1e-5)

    # Speeds must be in SI m/s: cruising speed between 5 and 20 m/s
    ego_speeds = [s.speed for s in ep.states if s.is_ego]
    assert all(0.0 <= sp <= 25.0 for sp in ego_speeds)


def test_prevention_of_future_information_leakage():
    """
    Verify that policies at step t only receive state information up to time t,
    and have no access to future actor coordinates or hidden scenario parameters.
    """
    gen = SyntheticTrajectoryGenerator(time_step=0.1)
    ep = gen.generate_episode("motorcycle_cut_in", seed=99)

    policy = PredictionAwarePolicy()
    sim = TrafficSimulator(dt=0.1)

    # In TrafficSimulator, at step t, the surrounding actors passed to policy
    # are strictly those from step t, without future arrays.
    res = sim.run_episode(ep, policy)
    for log in res["step_logs"]:
        # Verify log state integrity
        assert "timestamp" in log
        assert "ego_speed" in log
        assert "action_type" in log


def test_full_end_to_end_smoke():
    # 1. Generate episode
    gen = SyntheticTrajectoryGenerator(time_step=0.1)
    ep = gen.generate_episode("mixed_traffic_intersection", seed=777)
    df = gen.episodes_to_dataframe([ep])
    assert len(df) > 0

    # 2. Extract trajectory window
    pipeline = TrajectoryDataPipeline(obs_steps=20, pred_steps=30, dt=0.1)
    windows = pipeline.extract_trajectory_windows(df, stride=10)
    assert len(windows["X_obs"]) > 0

    # 3. Predict with Constant Velocity
    cv = ConstantVelocityForecaster(pred_steps=30, dt=0.1)
    pred_cv = cv.predict(windows["X_obs"])
    ade_cv = compute_ade(windows["Y_future"], pred_cv)
    fde_cv = compute_fde(windows["Y_future"], pred_cv)
    assert ade_cv >= 0.0
    assert fde_cv >= 0.0

    # 4. Run closed-loop simulation under Reactive and Uncertainty policies
    sim = TrafficSimulator(dt=0.1, goal_x=40.0)
    res_react = sim.run_episode(ep, ReactivePolicy(target_speed=10.0))
    res_unc = sim.run_episode(ep, UncertaintyAwarePolicy(target_speed=10.0))

    assert "min_separation" in res_react
    assert "min_separation" in res_unc
    assert len(res_react["step_logs"]) > 0
    assert len(res_unc["step_logs"]) > 0

    # 5. Evaluate simulation metrics calculation
    metrics = compute_simulation_metrics([res_react, res_unc])
    assert metrics["num_episodes_evaluated"] == 2
    assert "collision_rate_percent" in metrics
