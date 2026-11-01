"""
Comprehensive Benchmark Evaluator and Ablation Runner.
Benchmarks:
1. Forecasting: Constant Velocity vs Deterministic GRU vs Probabilistic Uncertainty GRU.
2. Simulation: Reactive vs Prediction-Aware (CV) vs Prediction-Aware (GRU) vs Uncertainty-Aware.
3. Component Ablations: Risk quantiles and proactive braking thresholds.
4. Failure Case Analysis and safety-efficiency trade-off records.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.constant_velocity import ConstantVelocityForecaster
from src.models.recurrent_forecaster import TrajectoryGRU, RecurrentForecasterTrainer
from src.models.uncertainty import UncertaintyTrajectoryGRU, UncertaintyForecasterTrainer
from src.simulation.policies import ReactivePolicy, PredictionAwarePolicy, UncertaintyAwarePolicy
from src.simulation.simulator import TrafficSimulator
from src.data.synthetic_generator import SyntheticTrajectoryGenerator, ScenarioEpisode
from src.evaluation.metrics import (
    compute_ade,
    compute_fde,
    evaluate_forecasting_by_group,
    compute_bivariate_gaussian_nll,
    compute_empirical_coverage,
    compute_simulation_metrics
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Orchestrates comparative benchmarking across forecasting models,
    simulation policies, and parameter ablations.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        checkpoints_dir: str = "outputs/checkpoints",
        metrics_dir: str = "outputs/metrics"
    ):
        self.data_dir = data_dir
        self.checkpoints_dir = checkpoints_dir
        self.metrics_dir = metrics_dir
        os.makedirs(self.metrics_dir, exist_ok=True)

    def evaluate_forecasting_models(self) -> Dict[str, Any]:
        """
        Evaluate all forecasting models on the unseen Test Split windows.
        """
        test_path = os.path.join(self.data_dir, "test_windows.npz")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test windows not found at {test_path}")

        test_data = np.load(test_path, allow_pickle=True)
        X_test_raw = test_data["X_obs"]            # (N, 20, 6) physical unscaled
        X_test_scaled = test_data["X_obs_scaled"]  # (N, 20, 6) standardized
        Y_test_true = test_data["Y_future"]        # (N, 30, 2) physical metric displacements
        meta_list = list(test_data["meta"])

        results = {}

        # 1. Constant Velocity Baseline
        cv_model = ConstantVelocityForecaster(pred_steps=30, dt=0.1)
        Y_pred_cv = cv_model.predict(X_test_raw)

        cv_ade = compute_ade(Y_test_true, Y_pred_cv)
        cv_fde = compute_fde(Y_test_true, Y_pred_cv)
        cv_groups = evaluate_forecasting_by_group(Y_test_true, Y_pred_cv, meta_list)

        results["constant_velocity"] = {
            "overall_ade_meters": round(cv_ade, 4),
            "overall_fde_meters": round(cv_fde, 4),
            "breakdown": cv_groups.to_dict(orient="records")
        }

        # 2. Deterministic GRU Forecaster
        det_ckpt = os.path.join(self.checkpoints_dir, "gru_deterministic_best.pt")
        if os.path.exists(det_ckpt):
            det_gru = TrajectoryGRU(input_dim=6, hidden_dim=64, num_layers=2, pred_steps=30, output_dim=2)
            det_trainer = RecurrentForecasterTrainer(model=det_gru)
            det_trainer.load_checkpoint(det_ckpt)

            Y_pred_gru = det_trainer.predict(X_test_scaled)
            gru_ade = compute_ade(Y_test_true, Y_pred_gru)
            gru_fde = compute_fde(Y_test_true, Y_pred_gru)
            gru_groups = evaluate_forecasting_by_group(Y_test_true, Y_pred_gru, meta_list)

            results["deterministic_gru"] = {
                "overall_ade_meters": round(gru_ade, 4),
                "overall_fde_meters": round(gru_fde, 4),
                "breakdown": gru_groups.to_dict(orient="records")
            }

        # 3. Uncertainty-Aware GRU Forecaster
        unc_ckpt = os.path.join(self.checkpoints_dir, "gru_uncertainty_best.pt")
        if os.path.exists(unc_ckpt):
            unc_gru = UncertaintyTrajectoryGRU(input_dim=6, hidden_dim=64, num_layers=2, pred_steps=30)
            unc_trainer = UncertaintyForecasterTrainer(model=unc_gru)
            unc_trainer.load_checkpoint(unc_ckpt)

            unc_preds = unc_trainer.predict(X_test_scaled)
            mu_pred = unc_preds["mu"]
            log_sig_pred = unc_preds["log_sigma"]
            rho_pred = unc_preds["rho"]

            unc_ade = compute_ade(Y_test_true, mu_pred)
            unc_fde = compute_fde(Y_test_true, mu_pred)
            unc_nll = compute_bivariate_gaussian_nll(Y_test_true, mu_pred, log_sig_pred, rho_pred)
            cov_95, area_95 = compute_empirical_coverage(Y_test_true, mu_pred, log_sig_pred, rho_pred, quantile=0.95)
            cov_68, area_68 = compute_empirical_coverage(Y_test_true, mu_pred, log_sig_pred, rho_pred, quantile=0.68)
            unc_groups = evaluate_forecasting_by_group(Y_test_true, mu_pred, meta_list)

            results["uncertainty_gru"] = {
                "overall_ade_meters": round(unc_ade, 4),
                "overall_fde_meters": round(unc_fde, 4),
                "negative_log_likelihood": round(unc_nll, 4),
                "coverage_95_percent": round(cov_95 * 100.0, 2),
                "mean_ellipse_area_95_m2": round(area_95, 3),
                "coverage_68_percent": round(cov_68 * 100.0, 2),
                "mean_ellipse_area_68_m2": round(area_68, 3),
                "breakdown": unc_groups.to_dict(orient="records")
            }

        # Save to disk
        out_json = os.path.join(self.metrics_dir, "forecasting_benchmark.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

    def evaluate_simulation_policies(
        self,
        num_episodes_per_scenario: int = 15,
        base_seed: int = 5000
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate matched closed-loop simulation episodes comparing:
        - Reactive Baseline
        - Prediction-Aware Policy (CV point forecasts)
        - Prediction-Aware Policy (GRU point forecasts)
        - Uncertainty-Aware Policy (Bivariate Gaussian covariance)
        """
        generator = SyntheticTrajectoryGenerator(time_step=0.1)
        scenarios = [
            "pedestrian_crossing",
            "motorcycle_cut_in",
            "mixed_traffic_intersection",
            "unregulated_overtaking"
        ]

        # Generate matched episodes
        episodes: List[ScenarioEpisode] = []
        cur_seed = base_seed
        for sc in scenarios:
            for _ in range(num_episodes_per_scenario):
                ep = generator.generate_episode(scenario_type=sc, seed=cur_seed)
                episodes.append(ep)
                cur_seed += 1

        sim = TrafficSimulator(dt=0.1, goal_x=45.0)

        policies = {
            "Reactive_Baseline": ReactivePolicy(target_speed=11.11),
            "Prediction_Aware_CV": PredictionAwarePolicy(target_speed=11.11),
            "Prediction_Aware_GRU": PredictionAwarePolicy(target_speed=11.11),
            "Uncertainty_Aware_Policy": UncertaintyAwarePolicy(target_speed=11.11, conservative_quantile=0.90)
        }

        policy_results = {}
        all_episode_logs = []
        failure_cases = []

        for p_name, policy in policies.items():
            ep_logs = []
            for ep in episodes:
                run_res = sim.run_episode(ep, policy)
                ep_logs.append(run_res)
                all_episode_logs.append(run_res)

                # Identify failure cases: collisions, near-collisions, or excessive delays
                if run_res["collision"] or run_res["near_collision"] or run_res["unnecessary_intervention"]:
                    failure_cases.append({
                        "policy": p_name,
                        "episode_id": ep.episode_id,
                        "scenario_type": ep.scenario_type,
                        "seed": ep.random_seed,
                        "outcome": "COLLISION" if run_res["collision"] else "NEAR_COLLISION",
                        "min_separation": run_res["min_separation"],
                        "elapsed_time": run_res["elapsed_time"],
                        "emergency_brakes": run_res["emergency_braking_count"],
                        "unnecessary_intervention": run_res["unnecessary_intervention"]
                    })

            metrics = compute_simulation_metrics(ep_logs)
            policy_results[p_name] = metrics

        # Save simulation metrics & failure cases
        sim_json = os.path.join(self.metrics_dir, "simulation_benchmark.json")
        with open(sim_json, "w", encoding="utf-8") as f:
            json.dump(policy_results, f, indent=2)

        fail_json = os.path.join(self.metrics_dir, "failure_cases.json")
        with open(fail_json, "w", encoding="utf-8") as f:
            json.dump(failure_cases, f, indent=2)

        return policy_results, failure_cases

    def run_ablation_study(self) -> Dict[str, Any]:
        """
        Ablation Study: Change one component at a time to examine:
        1. Impact of Uncertainty Quantile (tau = 0.70 vs 0.90 vs 0.99) on safety vs efficiency.
        2. Impact of Prediction Proactive Threshold (d = 1.5m vs 2.5m vs 4.0m).
        """
        generator = SyntheticTrajectoryGenerator(time_step=0.1)
        test_episodes = [
            generator.generate_episode("motorcycle_cut_in", seed=9001),
            generator.generate_episode("motorcycle_cut_in", seed=9002),
            generator.generate_episode("pedestrian_crossing", seed=9003),
            generator.generate_episode("pedestrian_crossing", seed=9004),
            generator.generate_episode("mixed_traffic_intersection", seed=9005),
            generator.generate_episode("mixed_traffic_intersection", seed=9006)
        ]

        sim = TrafficSimulator(dt=0.1, goal_x=45.0)
        ablation_results = {}

        # Ablation 1: Uncertainty Quantile
        for q in [0.70, 0.90, 0.99]:
            pol = UncertaintyAwarePolicy(target_speed=11.11, conservative_quantile=q)
            logs = [sim.run_episode(ep, pol) for ep in test_episodes]
            met = compute_simulation_metrics(logs)
            ablation_results[f"uncertainty_quantile_{int(q*100)}pct"] = met

        # Ablation 2: Proactive Distance Threshold
        for thresh in [1.5, 2.5, 4.0]:
            pol = PredictionAwarePolicy(target_speed=11.11, predicted_min_sep_crit=thresh)
            logs = [sim.run_episode(ep, pol) for ep in test_episodes]
            met = compute_simulation_metrics(logs)
            ablation_results[f"proactive_dist_threshold_{thresh}m"] = met

        abl_json = os.path.join(self.metrics_dir, "ablation_study.json")
        with open(abl_json, "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2)

        return ablation_results
