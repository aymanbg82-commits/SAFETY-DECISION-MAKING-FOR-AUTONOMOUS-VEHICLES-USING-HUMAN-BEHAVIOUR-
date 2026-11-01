"""
Unit tests for safety policies (Reactive, Prediction-Aware, Uncertainty-Aware)
and closed-loop traffic simulation.
"""

import pytest
import numpy as np
from src.simulation.policies import ReactivePolicy, PredictionAwarePolicy, UncertaintyAwarePolicy
from src.simulation.policies import CalibratedUncertaintyPolicy
from src.evaluation.uncertainty_calibration import (
    calibrate_cv_residual_envelope,
    evaluate_cv_residual_calibration,
)
from src.simulation.simulator import TrafficSimulator
from src.data.synthetic_generator import SyntheticTrajectoryGenerator


def test_reactive_policy_clear_corridor_cruises():
    policy = ReactivePolicy(target_speed=11.11)
    ego_state = {
        "step": 0, "timestamp": 0.0, "x": 0.0, "y": 0.0,
        "vx": 8.0, "vy": 0.0, "speed": 8.0, "heading": 0.0,
        "length": 4.5, "width": 2.0
    }
    action = policy.compute_action(ego_state, surrounding_actors=[])
    assert action["action_type"] == "cruising"
    assert action["acceleration"] > 0.0  # Should accelerate towards 11.11 m/s


def test_reactive_policy_critical_proximity_emergency_brakes():
    policy = ReactivePolicy(target_speed=11.11, emergency_dist=2.0)
    ego_state = {
        "step": 0, "timestamp": 0.0, "x": 0.0, "y": 0.0,
        "vx": 10.0, "vy": 0.0, "speed": 10.0, "heading": 0.0,
        "length": 4.5, "width": 2.0
    }
    # Actor directly 1.5m ahead
    hazard_actor = [{
        "actor_id": "ped_01", "actor_type": "pedestrian",
        "x": 3.5, "y": 0.0, "vx": 0.0, "vy": 0.0, "speed": 0.0,
        "heading": 0.0, "length": 0.6, "width": 0.6
    }]
    action = policy.compute_action(ego_state, hazard_actor)
    assert action["action_type"] == "emergency_braking"
    assert action["acceleration"] <= -6.0


def test_policy_action_limits_strictly_respected():
    policy = PredictionAwarePolicy()
    ego_state = {
        "step": 0, "timestamp": 0.0, "x": 0.0, "y": 0.0,
        "vx": 0.0, "vy": 0.0, "speed": 0.0, "heading": 0.0,
        "length": 4.5, "width": 2.0
    }
    action = policy.compute_action(ego_state, surrounding_actors=[])
    assert -7.0 <= action["acceleration"] <= 3.5


def test_simulator_closed_loop_execution():
    gen = SyntheticTrajectoryGenerator(time_step=0.1)
    episode = gen.generate_episode("pedestrian_crossing", seed=42)

    sim = TrafficSimulator(dt=0.1, goal_x=40.0)
    policy = ReactivePolicy()
    result = sim.run_episode(episode, policy)

    assert "collision" in result
    assert "near_collision" in result
    assert "task_completed" in result
    assert "min_separation" in result
    assert "elapsed_time" in result
    assert len(result["step_logs"]) > 0

    # Ensure kinematics logged properly
    for step in result["step_logs"]:
        assert step["ego_speed"] >= 0.0
        assert step["min_separation"] >= 0.0


def test_autorickshaw_cut_in_has_configured_lateral_motion():
    gen = SyntheticTrajectoryGenerator(time_step=0.1)
    episode = gen.generate_episode(
        "autorickshaw_cut_in",
        seed=42,
        custom_params={
            "ego_speed": 10.0,
            "cut_in_speed": 8.0,
            "cut_in_gap": 8.0,
            "lateral_offset": 2.5,
            "lateral_speed": 2.0,
            "cut_in_step": 10,
        },
    )
    auto_states = [state for state in episode.states if state.actor_id == "auto_01"]

    assert auto_states[0].y == pytest.approx(2.5)
    assert auto_states[9].vy == pytest.approx(0.0)
    assert auto_states[10].vy == pytest.approx(-2.0)
    assert episode.parameters["cut_in_gap"] == pytest.approx(8.0)


def test_policy_observation_contains_current_actor_only():
    class ObservationSpy(ReactivePolicy):
        def __init__(self):
            super().__init__()
            self.actor_keys = []

        def compute_action(self, ego_state, surrounding_actors):
            self.actor_keys.append(set(surrounding_actors[0].keys()) if surrounding_actors else set())
            return super().compute_action(ego_state, surrounding_actors)

    episode = SyntheticTrajectoryGenerator(time_step=0.1).generate_episode(
        "pedestrian_crossing", seed=9, custom_params={"crossing_time": 1.0}
    )
    spy = ObservationSpy()
    TrafficSimulator(dt=0.1).run_episode(episode, spy)

    assert spy.actor_keys
    assert all("predicted_future" not in keys for keys in spy.actor_keys)
    assert all("future_states" not in keys for keys in spy.actor_keys)


def test_animal_entry_lateral_motion_and_collision_geometry():
    generator = SyntheticTrajectoryGenerator(time_step=0.1)
    episode = generator.generate_episode(
        "animal_entry",
        seed=12,
        custom_params={
            "ego_speed": 10.0,
            "animal_start_distance": 18.0,
            "animal_entry_time": 0.5,
            "animal_direction": -1.0,
            "animal_speed": 4.0,
            "animal_length": 1.4,
            "animal_width": 0.8,
        },
    )
    animal_states = [state for state in episode.states if state.actor_id == "animal_01"]
    assert animal_states[0].x == pytest.approx(3.0, abs=0.1)
    assert animal_states[4].vy == pytest.approx(0.0)
    assert animal_states[5].vy == pytest.approx(-4.0)
    assert animal_states[5].length == pytest.approx(1.4)
    assert animal_states[5].width == pytest.approx(0.8)

    result = TrafficSimulator(dt=0.1, goal_x=45.0).run_episode(
        episode, PredictionAwarePolicy(target_speed=10.0)
    )
    assert result["min_separation"] >= 0.0
    assert all("animal_01" in [actor["actor_id"] for actor in step["surrounding_actors"]] for step in result["step_logs"])


def test_cv_uncertainty_calibration_uses_validation_residuals():
    x_obs = np.zeros((10, 20, 6), dtype=float)
    y_future = np.zeros((10, 30, 2), dtype=float)
    y_future[:, :, 0] = 1.0
    profile = calibrate_cv_residual_envelope(x_obs, y_future, quantile=0.9)
    metrics = evaluate_cv_residual_calibration(x_obs, y_future, profile)

    assert profile["method"] == "cv_residual_euclidean_quantile"
    assert profile["validation_sample_count"] == 10
    assert metrics["mean_marginal_coverage_percent"] >= 90.0
    assert metrics["final_horizon_coverage_percent"] >= 90.0
    assert "not joint trajectory coverage" in profile["coverage_semantics"]


def test_calibrated_uncertainty_policy_respects_action_limits():
    profile = {
        "quantile": 0.9,
        "horizons": [0.1, 0.2],
        "sigma_x": [0.5, 0.5],
        "sigma_y": [0.5, 0.5],
        "rho": [0.0, 0.0],
    }
    policy = CalibratedUncertaintyPolicy(profile, target_speed=10.0)
    ego = {
        "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "speed": 10.0,
        "heading": 0.0, "length": 4.5, "width": 2.0,
    }
    action = policy.compute_action(ego, [])
    assert -7.0 <= action["acceleration"] <= 3.5
