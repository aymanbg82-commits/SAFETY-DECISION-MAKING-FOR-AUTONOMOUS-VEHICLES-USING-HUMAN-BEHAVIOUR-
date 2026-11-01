"""
Executable script to generate synthetic traffic episodes, execute data pipeline,
perform zero-leakage splits, extract trajectory windows, and save artifacts.
"""

import os
import sys
import json
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.data.data_pipeline import TrajectoryDataPipeline


def main():
    print("=" * 70)
    print("GENERATING SYNTHETIC MULTI-AGENT TRAFFIC DATASET (INDIAN ROAD SCENARIOS)")
    print("=" * 70)

    # 1. Initialize Generator
    generator = SyntheticTrajectoryGenerator(time_step=0.1, default_seed=42)
    scenarios = [
        "pedestrian_crossing",
        "motorcycle_cut_in",
        "mixed_traffic_intersection",
        "unregulated_overtaking"
    ]
    episodes_per_scenario = 40
    print(f"Generating {episodes_per_scenario} episodes per scenario across {len(scenarios)} scenarios...")

    episodes = generator.generate_batch_dataset(
        num_episodes_per_scenario=episodes_per_scenario,
        scenarios=scenarios,
        base_seed=1000
    )
    df = generator.episodes_to_dataframe(episodes)
    print(f"Total raw records generated: {len(df):,} rows across {df['episode_id'].nunique()} episodes.")

    # Save raw synthetic table
    os.makedirs(os.path.join("data", "synthetic"), exist_ok=True)
    parquet_path = os.path.join("data", "synthetic", "synthetic_traffic_episodes.parquet")
    csv_sample_path = os.path.join("data", "synthetic", "synthetic_sample_1000.csv")
    df.to_parquet(parquet_path, index=False)
    df.head(1000).to_csv(csv_sample_path, index=False)
    print(f"Saved complete dataset to: {parquet_path}")

    # 2. Pipeline Validation & Cleaning
    pipeline = TrajectoryDataPipeline(obs_steps=20, pred_steps=30, dt=0.1)
    is_valid, issues = pipeline.validate_schema(df)
    print(f"Schema Validation Passed: {is_valid}")
    if not is_valid:
        print(f"Schema Issues: {issues}")
        return

    cleaned_df = pipeline.clean_and_filter(df)
    print(f"Exclusion Log: {pipeline.exclusion_log}")

    # 3. Zero-Leakage Split by Scene / Episode
    train_eps, val_eps, test_eps = pipeline.split_episodes(
        cleaned_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    print(f"Zero-Leakage Split Counts: Train={len(train_eps)}, Val={len(val_eps)}, Test={len(test_eps)} episodes.")

    # Verify no overlap between splits
    assert len(set(train_eps).intersection(set(val_eps))) == 0, "Leakage between train and val!"
    assert len(set(train_eps).intersection(set(test_eps))) == 0, "Leakage between train and test!"
    assert len(set(val_eps).intersection(set(test_eps))) == 0, "Leakage between val and test!"

    # 4. Extract Trajectory Windows
    train_windows = pipeline.extract_trajectory_windows(cleaned_df, episode_whitelist=train_eps, stride=5)
    val_windows = pipeline.extract_trajectory_windows(cleaned_df, episode_whitelist=val_eps, stride=5)
    test_windows = pipeline.extract_trajectory_windows(cleaned_df, episode_whitelist=test_eps, stride=5)

    print(f"Extracted Windows: Train={len(train_windows['X_obs'])}, Val={len(val_windows['X_obs'])}, Test={len(test_windows['X_obs'])}")

    # 5. Fit Scalers on Training Windows Only
    pipeline.fit_scalers(train_windows["X_obs"], train_windows["Y_future"])

    # Transform features
    X_train_scaled = pipeline.transform_features(train_windows["X_obs"])
    X_val_scaled = pipeline.transform_features(val_windows["X_obs"])
    X_test_scaled = pipeline.transform_features(test_windows["X_obs"])

    # 6. Save Processed Split Windows
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, "train_windows.npz"),
        X_obs=train_windows["X_obs"],
        X_obs_scaled=X_train_scaled,
        Y_future=train_windows["Y_future"],
        meta=train_windows["meta"]
    )
    np.savez_compressed(
        os.path.join(out_dir, "val_windows.npz"),
        X_obs=val_windows["X_obs"],
        X_obs_scaled=X_val_scaled,
        Y_future=val_windows["Y_future"],
        meta=val_windows["meta"]
    )
    np.savez_compressed(
        os.path.join(out_dir, "test_windows.npz"),
        X_obs=test_windows["X_obs"],
        X_obs_scaled=X_test_scaled,
        Y_future=test_windows["Y_future"],
        meta=test_windows["meta"]
    )

    manifest = {
        "train_episodes": train_eps,
        "val_episodes": val_eps,
        "test_episodes": test_eps,
        "train_windows_count": len(train_windows["X_obs"]),
        "val_windows_count": len(val_windows["X_obs"]),
        "test_windows_count": len(test_windows["X_obs"]),
        "obs_steps": 20,
        "pred_steps": 30,
        "dt": 0.1,
        "scenarios": scenarios
    }
    with open(os.path.join(out_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print("SYNTHETIC DATA PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
