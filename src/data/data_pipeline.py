"""
Data Pipeline: Schema validation, invalid-track filtering, exclusion counts,
zero-leakage scenario-based train/val/test splits, sliding-window generation,
and strictly training-fitted feature scaling.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "step", "timestamp", "actor_id", "actor_type",
    "x", "y", "vx", "vy", "speed", "heading",
    "length", "width", "is_ego", "episode_id",
    "scenario_type", "random_seed", "is_synthetic"
]


class TrajectoryDataPipeline:
    """
    Robust pipeline for trajectory data processing, validation,
    splitting, and window formatting.
    """

    def __init__(
        self,
        obs_steps: int = 20,
        pred_steps: int = 30,
        dt: float = 0.1,
        processed_dir: Optional[str] = None
    ):
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.total_window_steps = obs_steps + pred_steps
        self.dt = dt
        self.processed_dir = processed_dir or os.path.join("data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None
        self.exclusion_log: Dict[str, int] = {
            "initial_rows": 0,
            "nan_or_inf_rows": 0,
            "short_tracks_excluded": 0,
            "valid_episodes": 0,
            "valid_windows_extracted": 0
        }

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate schema compliance against required fields and data constraints.
        """
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            return False, [f"Missing required columns: {missing}"]

        issues = []
        if not np.issubdtype(df["x"].dtype, np.number):
            issues.append("Column 'x' is not numeric")
        if not np.issubdtype(df["y"].dtype, np.number):
            issues.append("Column 'y' is not numeric")
        if not np.issubdtype(df["timestamp"].dtype, np.number):
            issues.append("Column 'timestamp' is not numeric")

        return len(issues) == 0, issues

    def clean_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data, drop non-finite values, and filter out tracks shorter than window requirements.
        Updates self.exclusion_log with transparent exclusion counts.
        """
        self.exclusion_log["initial_rows"] = len(df)

        # Check for NaN / Inf
        finite_mask = np.isfinite(df[["x", "y", "vx", "vy", "speed", "heading"]]).all(axis=1)
        nan_inf_count = int((~finite_mask).sum())
        self.exclusion_log["nan_or_inf_rows"] = nan_inf_count

        cleaned = df[finite_mask].copy()

        # Filter tracks that do not have enough steps
        valid_tracks = []
        grouped = cleaned.groupby(["episode_id", "actor_id"])
        short_count = 0

        for (ep_id, act_id), track in grouped:
            if len(track) < self.total_window_steps:
                short_count += 1
            else:
                valid_tracks.append(track)

        self.exclusion_log["short_tracks_excluded"] = short_count

        if not valid_tracks:
            return pd.DataFrame(columns=df.columns)

        filtered_df = pd.concat(valid_tracks, ignore_index=True)
        self.exclusion_log["valid_episodes"] = filtered_df["episode_id"].nunique()
        return filtered_df

    def split_episodes(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split STRICTLY by episode/scene before generating overlapping windows.
        Stratified by scenario_type so all splits have balanced representation.
        Guarantees zero data leakage across train, val, and test splits.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
        rng = np.random.RandomState(seed)

        # Get unique episodes and their scenario type
        ep_meta = df[["episode_id", "scenario_type"]].drop_duplicates()
        train_eps, val_eps, test_eps = [], [], []

        for sc_type, group in ep_meta.groupby("scenario_type"):
            ep_list = group["episode_id"].tolist()
            rng.shuffle(ep_list)

            n_total = len(ep_list)
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * val_ratio))

            train_eps.extend(ep_list[:n_train])
            val_eps.extend(ep_list[n_train:n_train + n_val])
            test_eps.extend(ep_list[n_train + n_val:])

        return train_eps, val_eps, test_eps

    def extract_trajectory_windows(
        self,
        df: pd.DataFrame,
        episode_whitelist: Optional[List[str]] = None,
        stride: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Extract sliding trajectory windows:
        - X_obs: (N, obs_steps, num_features) relative to observation endpoint.
        - Y_future: (N, pred_steps, 2) metric displacements (dx, dy) from endpoint.
        - Meta: array of dicts storing episode_id, actor_id, actor_type, endpoint (x0, y0).
        """
        if episode_whitelist is not None:
            active_df = df[df["episode_id"].isin(episode_whitelist)].copy()
        else:
            active_df = df.copy()

        obs_list = []
        fut_list = []
        meta_list = []

        feature_cols = ["x", "y", "vx", "vy", "speed", "heading"]

        # Group by episode and actor to ensure window continuity
        for (ep_id, act_id), track in active_df.groupby(["episode_id", "actor_id"]):
            track = track.sort_values("step").reset_index(drop=True)
            n_steps = len(track)

            if n_steps < self.total_window_steps:
                continue

            actor_type = track["actor_type"].iloc[0]
            is_ego = bool(track["is_ego"].iloc[0])

            # Sliding window extraction
            for start_idx in range(0, n_steps - self.total_window_steps + 1, stride):
                obs_end_idx = start_idx + self.obs_steps
                pred_end_idx = obs_end_idx + self.pred_steps

                obs_window = track.iloc[start_idx:obs_end_idx]
                fut_window = track.iloc[obs_end_idx:pred_end_idx]

                # Endpoint reference frame (relative translation)
                ref_x = obs_window["x"].iloc[-1]
                ref_y = obs_window["y"].iloc[-1]

                # Relative observation features
                rel_obs = np.array(obs_window[feature_cols].values, dtype=np.float64, copy=True)
                rel_obs[:, 0] -= ref_x  # relative dx
                rel_obs[:, 1] -= ref_y  # relative dy

                # Future ground-truth displacements from ref
                fut_coords = np.array(fut_window[["x", "y"]].values, dtype=np.float64, copy=True)
                rel_future = fut_coords - np.array([ref_x, ref_y])

                obs_list.append(rel_obs)
                fut_list.append(rel_future)
                meta_list.append({
                    "episode_id": ep_id,
                    "actor_id": act_id,
                    "actor_type": actor_type,
                    "is_ego": is_ego,
                    "start_step": start_idx,
                    "ref_x": float(ref_x),
                    "ref_y": float(ref_y),
                    "scenario_type": track["scenario_type"].iloc[0]
                })

        N = len(obs_list)
        if N == 0:
            return {
                "X_obs": np.zeros((0, self.obs_steps, len(feature_cols))),
                "Y_future": np.zeros((0, self.pred_steps, 2)),
                "meta": []
            }

        return {
            "X_obs": np.array(obs_list, dtype=np.float32),
            "Y_future": np.array(fut_list, dtype=np.float32),
            "meta": meta_list
        }

    def fit_scalers(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Fit feature and target scalers strictly on the training set.
        """
        N, T_obs, F = X_train.shape
        self.feature_scaler = StandardScaler()
        # Reshape to 2D for standard scaling across all timesteps
        self.feature_scaler.fit(X_train.reshape(-1, F))

        N_fut, T_pred, D = Y_train.shape
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(Y_train.reshape(-1, D))

        # Save scaler parameters to disk
        scaler_data = {
            "feature_mean": self.feature_scaler.mean_.tolist(),
            "feature_scale": self.feature_scaler.scale_.tolist(),
            "target_mean": self.target_scaler.mean_.tolist(),
            "target_scale": self.target_scaler.scale_.tolist()
        }
        out_path = os.path.join(self.processed_dir, "scaler_params.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scaler_data, f, indent=2)

        logger.info(f"Fitted scalers saved to {out_path}")

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform observation features using the training-fitted scaler."""
        if self.feature_scaler is None:
            raise RuntimeError("Feature scaler has not been fitted.")
        N, T, F = X.shape
        X_scaled = self.feature_scaler.transform(X.reshape(-1, F))
        return X_scaled.reshape(N, T, F)

    def inverse_transform_targets(self, Y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform predicted targets back to physical SI metres."""
        if self.target_scaler is None:
            raise RuntimeError("Target scaler has not been fitted.")
        N, T, D = Y_scaled.shape
        Y_inv = self.target_scaler.inverse_transform(Y_scaled.reshape(-1, D))
        return Y_inv.reshape(N, T, D)
