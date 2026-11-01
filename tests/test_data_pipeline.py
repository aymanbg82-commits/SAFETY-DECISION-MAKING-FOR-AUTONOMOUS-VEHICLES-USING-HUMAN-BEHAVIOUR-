"""
Unit tests for data pipeline, schema validation, zero-leakage splits,
and reproducible synthetic scenario generation.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.data.data_pipeline import TrajectoryDataPipeline


def test_synthetic_generator_deterministic_seed():
    gen = SyntheticTrajectoryGenerator(time_step=0.1, default_seed=42)
    ep1 = gen.generate_episode("pedestrian_crossing", seed=123)
    ep2 = gen.generate_episode("pedestrian_crossing", seed=123)

    assert ep1.episode_id == ep2.episode_id
    assert len(ep1.states) == len(ep2.states)
    assert ep1.parameters == ep2.parameters

    # Positions must be strictly identical for the same seed
    for s1, s2 in zip(ep1.states, ep2.states):
        assert s1.x == pytest.approx(s2.x, abs=1e-6)
        assert s1.y == pytest.approx(s2.y, abs=1e-6)


def test_schema_validation_catches_missing_columns():
    pipeline = TrajectoryDataPipeline()
    # Missing 'heading' and 'width'
    df_bad = pd.DataFrame({
        "step": [0], "timestamp": [0.0], "actor_id": ["ego"],
        "actor_type": ["car"], "x": [0.0], "y": [0.0],
        "vx": [10.0], "vy": [0.0], "speed": [10.0],
        "length": [4.5], "is_ego": [True], "episode_id": ["ep1"],
        "scenario_type": ["pedestrian_crossing"], "random_seed": [42],
        "is_synthetic": [True]
    })
    is_valid, issues = pipeline.validate_schema(df_bad)
    assert is_valid is False
    assert any("Missing required columns" in issue for issue in issues)


def test_zero_leakage_split_isolation():
    gen = SyntheticTrajectoryGenerator(time_step=0.1)
    episodes = gen.generate_batch_dataset(num_episodes_per_scenario=10, base_seed=500)
    df = gen.episodes_to_dataframe(episodes)

    pipeline = TrajectoryDataPipeline()
    train_eps, val_eps, test_eps = pipeline.split_episodes(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    set_train = set(train_eps)
    set_val = set(val_eps)
    set_test = set(test_eps)

    # Hard zero-leakage guarantee
    assert len(set_train.intersection(set_val)) == 0, "Leakage between train and val!"
    assert len(set_train.intersection(set_test)) == 0, "Leakage between train and test!"
    assert len(set_val.intersection(set_test)) == 0, "Leakage between val and test!"


def test_sliding_window_dimensions_and_scaling():
    gen = SyntheticTrajectoryGenerator(time_step=0.1)
    episodes = [
        gen.generate_episode("pedestrian_crossing", seed=10),
        gen.generate_episode("motorcycle_cut_in", seed=20)
    ]
    df = gen.episodes_to_dataframe(episodes)

    pipeline = TrajectoryDataPipeline(obs_steps=20, pred_steps=30, dt=0.1)
    windows = pipeline.extract_trajectory_windows(df, stride=5)

    X_obs = windows["X_obs"]
    Y_fut = windows["Y_future"]

    assert X_obs.ndim == 3
    assert X_obs.shape[1] == 20  # 20 observation steps
    assert X_obs.shape[2] == 6   # 6 features
    assert Y_fut.ndim == 3
    assert Y_fut.shape[1] == 30  # 30 prediction steps
    assert Y_fut.shape[2] == 2   # (dx, dy)

    # Test fitting scaler and inverse transformation
    pipeline.fit_scalers(X_obs, Y_fut)
    X_scaled = pipeline.transform_features(X_obs)
    assert X_scaled.shape == X_obs.shape

    Y_restored = pipeline.inverse_transform_targets(
        pipeline.target_scaler.transform(Y_fut.reshape(-1, 2)).reshape(Y_fut.shape)
    )
    np.testing.assert_allclose(Y_fut, Y_restored, atol=1e-4)
