"""Focused contracts for the Streamlit-to-renderer integration."""

import json

import pytest

from app import run_config, validate_config


def _config(**overrides):
    config = {
        "scenario": "pedestrian_crossing",
        "speed_kmh": 36.0,
        "ped_speed": 1.2,
        "ped_start_distance": 25.0,
        "crossing_time": 1.0,
        "cut_in_step": 15,
        "hesitation": True,
        "policy": "prediction",
        "policy_label": "Point-prediction-aware (CV)",
        "seed": 42,
    }
    config.update(overrides)
    return config


def test_dashboard_rejects_unsupported_values():
    valid, message = validate_config(_config(speed_kmh=70.0))
    assert not valid
    assert "speed" in message


def test_renderer_payload_preserves_recorded_time_and_issue_alignment():
    result, payload, _ = run_config(json.dumps(_config(), sort_keys=True))
    timestamps = [frame["timestamp"] for frame in payload["frames"]]

    assert payload["coordinate_mapping"]["unit"] == "metres (SI)"
    assert timestamps == sorted(timestamps)
    assert payload["scenario_config"]["total_frames"] == len(result["step_logs"])
    for frame in payload["frames"]:
        assert frame["timestamp"] == pytest.approx(frame["step"] * 0.1, abs=1e-6)
        for prediction in frame["predictions"].values():
            assert prediction["horizons"][0] > 0
            assert prediction["horizons"][-1] <= 2.0


def test_same_configuration_is_reproducible():
    config = json.dumps(_config(seed=99), sort_keys=True)
    first, _, _ = run_config(config)
    second, _, _ = run_config(config)
    assert first["step_logs"] == second["step_logs"]


def test_pedestrian_controls_are_recorded_and_delay_crossing():
    result, _, parameters = run_config(json.dumps(_config(), sort_keys=True))

    assert parameters["ped_start_distance"] == pytest.approx(25.0)
    assert parameters["crossing_time"] == pytest.approx(1.0)
    pedestrian_logs = [
        log["surrounding_actors"][0]
        for log in result["step_logs"]
        if log["surrounding_actors"]
    ]
    assert pedestrian_logs[0]["x"] == pytest.approx(-15.0 + 25.0, abs=0.1)
    assert all(actor["vy"] == pytest.approx(0.0) for actor in pedestrian_logs[:10])
    assert pedestrian_logs[10]["vy"] > 0.0


def test_autorickshaw_cut_in_controls_reach_renderer_and_outcome():
    config = _config(
        scenario="autorickshaw_cut_in",
        policy="prediction",
        policy_label="Point-prediction-aware (CV)",
        cut_in_gap=8.0,
        cut_in_speed_kmh=28.0,
        cut_in_lateral_speed=2.0,
        cut_in_lateral_offset=2.5,
        cut_in_step=12,
    )
    result, payload, parameters = run_config(json.dumps(config, sort_keys=True))

    assert parameters["cut_in_gap"] == pytest.approx(8.0)
    assert parameters["cut_in_speed"] == pytest.approx(28.0 / 3.6)
    assert parameters["lateral_speed"] == pytest.approx(2.0)
    assert payload["actors"][1]["type"] == "autorickshaw"
    assert any(frame["predictions"]["auto_01"]["points"] for frame in payload["frames"])
    assert all(-7.0 <= log["acceleration"] <= 3.5 for log in result["step_logs"])
    assert "collision" in payload["summary"]


def test_matched_policies_share_initial_observation_and_action_limits():
    reactive = _config(policy="reactive", policy_label="Reactive baseline")
    prediction = _config(policy="prediction", policy_label="Point-prediction-aware (CV)")
    first_result, _, first_params = run_config(json.dumps(reactive, sort_keys=True))
    second_result, _, second_params = run_config(json.dumps(prediction, sort_keys=True))

    assert first_params == second_params
    assert first_result["step_logs"][0]["surrounding_actors"] == second_result["step_logs"][0]["surrounding_actors"]
    assert all(-7.0 <= x["acceleration"] <= 3.5 for x in first_result["step_logs"])
    assert all(-7.0 <= x["acceleration"] <= 3.5 for x in second_result["step_logs"])


def test_animal_entry_validation_and_renderer_payload():
    config = _config(
        scenario="animal_entry",
        policy="prediction",
        animal_start_distance=22.0,
        animal_entry_time=1.2,
        animal_direction=-1.0,
        animal_speed=2.5,
        animal_length=1.4,
        animal_width=0.8,
    )
    valid, message = validate_config(config)
    assert valid, message
    invalid, message = validate_config(dict(config, animal_width=3.0))
    assert not invalid
    assert "dimensions" in message

    result, payload, parameters = run_config(json.dumps(config, sort_keys=True))
    assert parameters["animal_start_distance"] == pytest.approx(22.0)
    assert parameters["animal_entry_time"] == pytest.approx(1.2)
    assert payload["actors"][1]["type"] == "animal"
    assert any(frame["predictions"]["animal_01"]["points"] for frame in payload["frames"])
    assert payload["scenario_config"]["scenario_type"] == "animal_entry"
    assert all(-7.0 <= step["acceleration"] <= 3.5 for step in result["step_logs"])


def test_animal_entry_is_reproducible():
    config = _config(scenario="animal_entry", animal_entry_time=2.0, seed=101)
    first = run_config(json.dumps(config, sort_keys=True))[0]
    second = run_config(json.dumps(config, sort_keys=True))[0]
    assert first["step_logs"] == second["step_logs"]


def test_calibrated_uncertainty_policy_payload_is_labeled():
    config = _config(
        scenario="animal_entry",
        policy="calibrated_uncertainty",
        policy_label="Calibrated uncertainty-aware (90% CV residual envelope)",
        seed=1201,
    )
    result, payload, _ = run_config(json.dumps(config, sort_keys=True))

    assert result["policy_name"] == "CalibratedUncertaintyPolicy"
    assert "validation-calibrated" in payload["scenario_config"]["uncertainty_label"]
    assert all(
        "validation-calibrated" in prediction["uncertainty_label"]
        for frame in payload["frames"]
        for prediction in frame["predictions"].values()
    )