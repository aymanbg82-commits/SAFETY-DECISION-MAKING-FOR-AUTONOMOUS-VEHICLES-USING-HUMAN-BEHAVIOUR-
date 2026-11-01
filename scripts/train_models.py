"""
Training script for Deterministic GRU and Uncertainty-Aware Bivariate Gaussian GRU.
Loads processed split windows, trains models with early stopping, and saves checkpoints.
"""

import os
import sys
import json
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.models.recurrent_forecaster import TrajectoryGRU, RecurrentForecasterTrainer
from src.models.uncertainty import UncertaintyTrajectoryGRU, UncertaintyForecasterTrainer


def main():
    print("=" * 70)
    print("TRAINING DETERMINISTIC & UNCERTAINTY-AWARE TRAJECTORY FORECASTERS")
    print("=" * 70)

    # 1. Load Processed Windows
    data_dir = os.path.join("data", "processed")
    train_npz = np.load(os.path.join(data_dir, "train_windows.npz"), allow_pickle=True)
    val_npz = np.load(os.path.join(data_dir, "val_windows.npz"), allow_pickle=True)

    X_train = train_npz["X_obs_scaled"]
    Y_train = train_npz["Y_future"]

    X_val = val_npz["X_obs_scaled"]
    Y_val = val_npz["Y_future"]

    print(f"Loaded Train Dataset: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Loaded Val Dataset:   X={X_val.shape}, Y={Y_val.shape}")

    checkpoints_dir = os.path.join("outputs", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 2. Train Deterministic GRU Forecaster
    print("\n--- [1/2] Training Deterministic GRU Forecaster (MSE Loss) ---")
    det_gru = TrajectoryGRU(input_dim=6, hidden_dim=64, num_layers=2, pred_steps=30, output_dim=2)
    det_trainer = RecurrentForecasterTrainer(model=det_gru, lr=0.001)

    det_ckpt_path = os.path.join(checkpoints_dir, "gru_deterministic_best.pt")
    det_history = det_trainer.fit(
        X_train, Y_train, X_val, Y_val,
        epochs=25, batch_size=64, patience=6, checkpoint_path=det_ckpt_path
    )
    print(f"Deterministic GRU Training Complete. Best Val MSE: {min(det_history['val_loss']):.4f} m^2")
    print(f"Checkpoint saved to: {det_ckpt_path}")

    # 3. Train Uncertainty-Aware Bivariate Gaussian GRU Forecaster
    print("\n--- [2/2] Training Uncertainty Bivariate Gaussian GRU (NLL Loss) ---")
    unc_gru = UncertaintyTrajectoryGRU(input_dim=6, hidden_dim=64, num_layers=2, pred_steps=30)
    unc_trainer = UncertaintyForecasterTrainer(model=unc_gru, lr=0.001)

    unc_ckpt_path = os.path.join(checkpoints_dir, "gru_uncertainty_best.pt")
    unc_history = unc_trainer.fit(
        X_train, Y_train, X_val, Y_val,
        epochs=25, batch_size=64, patience=6, checkpoint_path=unc_ckpt_path
    )
    print(f"Uncertainty GRU Training Complete. Best Val NLL: {min(unc_history['val_nll']):.4f}")
    print(f"Checkpoint saved to: {unc_ckpt_path}")

    # 4. Save Training History
    metrics_dir = os.path.join("outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    training_summary = {
        "deterministic_gru": {
            "checkpoint": det_ckpt_path,
            "best_val_loss_mse": min(det_history["val_loss"]),
            "epochs_trained": len(det_history["train_loss"]),
            "final_train_loss": det_history["train_loss"][-1],
            "history": det_history
        },
        "uncertainty_gru": {
            "checkpoint": unc_ckpt_path,
            "best_val_loss_nll": min(unc_history["val_nll"]),
            "epochs_trained": len(unc_history["train_nll"]),
            "final_train_nll": unc_history["train_nll"][-1],
            "history": unc_history
        }
    }
    with open(os.path.join(metrics_dir, "training_records.json"), "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
