"""Calibrate and evaluate the CV residual envelope on validation windows."""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.uncertainty_calibration import (
    calibrate_cv_residual_envelope,
    evaluate_cv_residual_calibration,
)


def main() -> None:
    data = np.load("data/processed/val_windows.npz", allow_pickle=True)
    profile = calibrate_cv_residual_envelope(data["X_obs"], data["Y_future"], quantile=0.90)
    profile["validation_metrics"] = evaluate_cv_residual_calibration(data["X_obs"], data["Y_future"], profile)
    output = "outputs/metrics/uncertainty_calibration.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
    print(json.dumps({"output": output, "validation_metrics": profile["validation_metrics"]}, indent=2))


if __name__ == "__main__":
    main()