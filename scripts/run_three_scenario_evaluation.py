"""Evaluate reactive and prediction-aware policies on three matched scenarios.

This command intentionally excludes uncertainty handling and model training. It
uses fixed validation and final-test seeds, does not tune thresholds, and saves
episode-level configurations alongside aggregate synthetic outcomes.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.evaluation.metrics import compute_simulation_metrics
from src.simulation.policies import CalibratedUncertaintyPolicy, PredictionAwarePolicy, ReactivePolicy
from src.simulation.simulator import TrafficSimulator


SCENARIOS = {
    "pedestrian_crossing": {
        "ego_speed": 10.0, "ped_speed": 1.2, "ped_start_distance": 25.0,
        "crossing_time": 1.0, "has_hesitation": False,
    },
    "autorickshaw_cut_in": {
        "ego_speed": 10.0, "cut_in_gap": 8.0, "cut_in_speed": 7.8,
        "lateral_speed": 2.0, "lateral_offset": 2.5, "cut_in_step": 15,
    },
    "animal_entry": {
        "ego_speed": 10.0, "animal_start_distance": 25.0,
        "animal_entry_time": 1.5, "animal_direction": -1.0,
        "animal_speed": 2.0, "animal_length": 1.2, "animal_width": 0.7,
    },
}


def run_evaluation(seeds: List[int]) -> Dict[str, Any]:
    generator = SyntheticTrajectoryGenerator(time_step=0.1)
    simulator = TrafficSimulator(dt=0.1, goal_x=45.0)
    policy_factories = {
        "reactive": lambda: ReactivePolicy(target_speed=10.0),
        "prediction_aware": lambda: PredictionAwarePolicy(target_speed=10.0),
        "calibrated_uncertainty": lambda: CalibratedUncertaintyPolicy(_load_calibration(), target_speed=10.0),
    }
    episodes: List[Dict[str, Any]] = []
    aggregate: Dict[str, Dict[str, Dict[str, Any]]] = {}
    failures: List[Dict[str, Any]] = []

    for scenario, params in SCENARIOS.items():
        aggregate[scenario] = {}
        for policy_name, make_policy in policy_factories.items():
            logs = []
            for seed in seeds:
                episode = generator.generate_episode(scenario, seed=seed, custom_params=params)
                result = simulator.run_episode(episode, make_policy())
                logs.append(result)
                episodes.append({
                    "scenario": scenario,
                    "seed": seed,
                    "policy": policy_name,
                    "configuration": params,
                    "metrics": {key: result[key] for key in (
                        "collision", "near_collision", "min_separation",
                        "emergency_braking_count", "task_completed",
                        "unnecessary_intervention", "elapsed_time"
                    )},
                })
            aggregate[scenario][policy_name] = compute_simulation_metrics(logs)

        for seed in seeds:
            reactive = next(x for x in episodes if x["scenario"] == scenario and x["seed"] == seed and x["policy"] == "reactive")
            predicted = next(x for x in episodes if x["scenario"] == scenario and x["seed"] == seed and x["policy"] == "prediction_aware")
            uncertainty = next(x for x in episodes if x["scenario"] == scenario and x["seed"] == seed and x["policy"] == "calibrated_uncertainty")
            if predicted["metrics"]["min_separation"] < reactive["metrics"]["min_separation"] or predicted["metrics"]["emergency_braking_count"] > reactive["metrics"]["emergency_braking_count"] or (not predicted["metrics"]["task_completed"] and reactive["metrics"]["task_completed"]):
                failures.append({
                    "scenario": scenario,
                    "seed": seed,
                    "type": "prediction_worse_or_more_braking",
                    "reactive": reactive["metrics"],
                    "prediction_aware": predicted["metrics"],
                })
            if uncertainty["metrics"]["min_separation"] < reactive["metrics"]["min_separation"] or uncertainty["metrics"]["emergency_braking_count"] > reactive["metrics"]["emergency_braking_count"] or (not uncertainty["metrics"]["task_completed"] and reactive["metrics"]["task_completed"]):
                failures.append({
                    "scenario": scenario,
                    "seed": seed,
                    "type": "uncertainty_worse_or_more_braking",
                    "reactive": reactive["metrics"],
                    "calibrated_uncertainty": uncertainty["metrics"],
                })
            if predicted["metrics"]["unnecessary_intervention"]:
                failures.append({
                    "scenario": scenario,
                    "seed": seed,
                    "type": "prediction_unnecessary_intervention",
                    "metrics": predicted["metrics"],
                })
            if uncertainty["metrics"]["unnecessary_intervention"]:
                failures.append({
                    "scenario": scenario,
                    "seed": seed,
                    "type": "uncertainty_unnecessary_intervention",
                    "metrics": uncertainty["metrics"],
                })

    return {
        "label": "SYNTHETIC_INDIAN_ROAD_SIMULATION",
        "evaluation_type": "aggregate_matched_policy_evaluation",
        "threshold_tuning": "none; existing policy thresholds were used unchanged",
        "validation_seeds": seeds,
        "final_test_seeds": seeds,
        "episode_count": len(episodes),
        "scenario_configurations": SCENARIOS,
        "aggregate_metrics": aggregate,
        "failure_cases": failures,
        "episodes": episodes,
    }


def _load_calibration() -> Dict[str, Any]:
    with open("outputs/metrics/uncertainty_calibration.json", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[1101, 1102])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[1201, 1202, 1203])
    parser.add_argument("--output", default="outputs/metrics/three_scenario_evaluation.json")
    args = parser.parse_args()
    validation = run_evaluation(args.validation_seeds)
    final_test = run_evaluation(args.test_seeds)
    results = {
        "label": "SYNTHETIC_INDIAN_ROAD_SIMULATION",
        "evaluation_type": "aggregate_matched_policy_evaluation",
        "threshold_tuning": "none; existing policy thresholds were used unchanged",
        "validation_seeds": args.validation_seeds,
        "final_test_seeds": args.test_seeds,
        "episode_count": validation["episode_count"] + final_test["episode_count"],
        "scenario_configurations": SCENARIOS,
        "validation": validation,
        "final_test": final_test,
        "aggregate_metrics": final_test["aggregate_metrics"],
        "failure_cases": final_test["failure_cases"],
        "episodes": validation["episodes"] + final_test["episodes"],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps({"output": args.output, "episode_count": results["episode_count"], "failure_count": len(results["failure_cases"])}, indent=2))


if __name__ == "__main__":
    main()