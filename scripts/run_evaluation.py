"""
Master Evaluation & Benchmarking Runner.
Runs end-to-end evaluations across forecasting models, simulation safety policies,
and component ablations. Saves metrics to outputs/metrics/.
"""

import os
import sys
import json
import pandas as pd

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.evaluation.evaluator import BenchmarkEvaluator


def main():
    print("=" * 70)
    print("RUNNING END-TO-END RESEARCH EVALUATION PIPELINE")
    print("=" * 70)

    evaluator = BenchmarkEvaluator()

    # 1. Trajectory Forecasting Benchmarks
    print("\n[1/3] Benchmarking Trajectory Forecasting Models on Unseen Test Split...")
    forecast_results = evaluator.evaluate_forecasting_models()

    print("\n--- FORECASTING SUMMARY (SI METRES) ---")
    for model_name, res in forecast_results.items():
        ade = res["overall_ade_meters"]
        fde = res["overall_fde_meters"]
        print(f"[{model_name.upper():22s}] ADE: {ade:.3f} m | FDE: {fde:.3f} m", end="")
        if "negative_log_likelihood" in res:
            nll = res["negative_log_likelihood"]
            cov95 = res["coverage_95_percent"]
            area95 = res["mean_ellipse_area_95_m2"]
            print(f" | NLL: {nll:.2f} | 95% Cov: {cov95}% (Area: {area95:.1f} m²)", end="")
        print()

    # 2. Closed-Loop Simulation Safety & Efficiency Benchmarks
    print("\n[2/3] Benchmarking Closed-Loop Safety Policies on Matched Episodes...")
    sim_results, failure_cases = evaluator.evaluate_simulation_policies(num_episodes_per_scenario=10, base_seed=7000)

    print("\n--- SIMULATION SAFETY & EFFICIENCY SUMMARY ---")
    headers = [
        "Policy", "Collisions", "Coll/1000", "Near-Coll %",
        "Min Sep (m)", "Emerg Actions", "Task Compl %", "Elapsed (s)"
    ]
    print(f"{headers[0]:<26} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<11} | {headers[4]:<11} | {headers[5]:<13} | {headers[6]:<12} | {headers[7]:<11}")
    print("-" * 125)

    for p_name, res in sim_results.items():
        print(
            f"{p_name:<26} | "
            f"{res['collision_count']:<10} | "
            f"{res['collisions_per_1000_episodes']:<10.1f} | "
            f"{res['near_collision_rate_percent']:<10.1f}% | "
            f"{res['mean_minimum_separation_meters']:<11.2f} | "
            f"{res['mean_emergency_actions_per_ep']:<13.2f} | "
            f"{res['task_completion_rate_percent']:<11.1f}% | "
            f"{res['mean_elapsed_time_seconds']:<11.2f}"
        )

    print(f"\nTotal failure cases / near-collisions logged: {len(failure_cases)}")

    # 3. Component Ablation Studies
    print("\n[3/3] Running Component Ablation Studies...")
    ablation_results = evaluator.run_ablation_study()

    print("\n--- ABLATION RESULTS ---")
    for abl_name, met in ablation_results.items():
        print(f"[{abl_name:<30}] Min Sep: {met['mean_minimum_separation_meters']:.2f}m | Collisions: {met['collision_count']} | Elapsed: {met['mean_elapsed_time_seconds']:.1f}s")

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS AND ABLATIONS COMPLETED AND RECORDED!")
    print("Outputs written to: outputs/metrics/")
    print("=" * 70)


if __name__ == "__main__":
    main()
