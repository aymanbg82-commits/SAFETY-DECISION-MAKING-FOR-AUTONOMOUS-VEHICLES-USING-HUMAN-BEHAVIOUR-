from scripts.run_three_scenario_evaluation import run_evaluation


def test_three_scenario_evaluation_has_matched_policy_metrics():
    results = run_evaluation([1201])

    assert results["episode_count"] == 9
    assert set(results["aggregate_metrics"]) == {
        "pedestrian_crossing",
        "autorickshaw_cut_in",
        "animal_entry",
    }
    for scenario_metrics in results["aggregate_metrics"].values():
        assert set(scenario_metrics) == {"reactive", "prediction_aware", "calibrated_uncertainty"}
        for metrics in scenario_metrics.values():
            assert metrics["num_episodes_evaluated"] == 1
            assert "collision_rate_percent" in metrics
            assert "near_collision_rate_percent" in metrics
            assert "mean_minimum_separation_meters" in metrics
            assert "mean_emergency_actions_per_ep" in metrics
            assert "task_completion_rate_percent" in metrics
            assert "mean_elapsed_time_seconds" in metrics