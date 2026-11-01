# Project Evidence Package

## Evidence scope

This report is based only on source files, configuration, saved outputs, and the test run performed during this inspection. A statement marked **verified** is supported by a saved artifact or by code executed during this inspection. A statement marked **configured** describes repository code or configuration and is not evidence that a separate run occurred.

## 1. Implementation Details

### Datasets and data use

- **Synthetic trajectory data - verified and used for model/evaluation artifacts.** `scripts/generate_synthetic_data.py` creates 40 episodes for each of four scenarios, using `base_seed=1000`, a 0.1 second timestep, and a 6 second duration. The configured split uses a 70/15/15 episode split with seed 42 and no episode overlap. The saved manifest reports 840 train windows, 180 validation windows, and 180 test windows, with 20 observation steps and 30 prediction steps.
- **Indian Road Dataset - audit/inspiration only.** The repository documents `thirdeyelabs/indian-road-dataset` and saves an audit JSON, but the raw 460.9 GB dataset is not present in this repository. The audit is used for taxonomy, scenario inspiration, and a justification for not converting uncalibrated pixel boxes into metric ground truth. It is not used as the trajectory-training dataset in the saved pipeline.
- **Processed arrays and synthetic tables.** Local saved artifacts include `data/processed/train_windows.npz`, `val_windows.npz`, `test_windows.npz`, `split_manifest.json`, and synthetic sample/table artifacts where present. The complete raw/generated tables and model checkpoints are excluded by `.gitignore` from the GitHub commit, so their presence in a local checkout should not be assumed from the Git history.

### Python and libraries

The README advertises Python 3.10+. No lock file or runtime version record is present. `requirements.txt` specifies:

- NumPy >=2.0, pandas >=2.2, scikit-learn >=1.4, SciPy >=1.12
- Matplotlib >=3.8, Plotly >=5.20, Streamlit >=1.35
- PyTorch >=2.2, pytest >=8.0, requests >=2.31
- PyYAML >=6.0, PyArrow >=15.0, tqdm >=4.66

### Technologies

- **Simulation:** Python, NumPy, SI metre/second coordinates, explicit 0.1 second simulation steps, and a closed-loop 2D traffic simulator.
- **ML/DL:** PyTorch GRU sequence models with Adam, MSE or bivariate Gaussian negative log-likelihood, validation checkpoints, and early stopping.
- **Evaluation:** NumPy/pandas metric aggregation, ADE/FDE, NLL, empirical Gaussian coverage, simulation safety metrics, ablation outputs, and JSON artifact storage.
- **Website:** Streamlit with Plotly charts and HTML components. The active dashboard has Overview, Scenario Lab, Compare Methods, Data Explorer, and Results and Failure Cases sections.
- **3D:** Local Three.js and OrbitControls embedded in generated HTML. The renderer creates a procedural road, actor meshes, trajectories, uncertainty ellipses, camera modes, playback, and telemetry HUD.

### Models and risk methods

- **Constant velocity baseline - implemented and benchmarked.** It extrapolates the endpoint `vx` and `vy` over 30 future steps.
- **Deterministic recurrent model - implemented and benchmarked.** `TrajectoryGRU` is a two-layer GRU with hidden dimension 64, six input features, dropout 0.1, and a 30-step, two-coordinate output head. It is trained with MSE.
- **Uncertainty model - implemented and benchmarked.** `UncertaintyTrajectoryGRU` is a two-layer GRU that emits `mu_x`, `mu_y`, `log_sigma_x`, `log_sigma_y`, and `rho` per future step. It uses a bivariate Gaussian NLL. The saved calibration profile instead uses a 90th-percentile empirical constant-velocity residual envelope from validation data and explicitly says its coverage is marginal per horizon, not joint trajectory coverage.
- **LSTM - not implemented.** The configuration comment says GRU or LSTM, but the source and saved checkpoints show GRU implementations only.
- **Risk classifier - not present.** There is no trained classification model. Risk is computed geometrically using oriented bounding boxes, Separating Axis Theorem collision checks, separation distance, TTC, predicted clearance, and an uncertainty-augmented clearance score.

### Simulation details

- **Scenarios:** pedestrian crossing, motorcycle cut-in, autorickshaw cut-in, sudden animal entry, mixed-traffic intersection, and unregulated overtaking are supported by the generator/dashboard. The main four-scenario dataset and benchmark use pedestrian crossing, motorcycle cut-in, mixed-traffic intersection, and unregulated overtaking. The separate three-scenario artifact uses pedestrian crossing, autorickshaw cut-in, and animal entry.
- **Actors:** cars, motorcycles, autorickshaws, trucks, pedestrians, and synthetic animals are represented with explicit dimensions. The main generated scenarios include ego cars plus pedestrians, motorcycles, autorickshaws, and oncoming/lead cars as applicable.
- **Dynamics:** parameterized kinematic trajectories in a planar metric world. The simulator updates ego longitudinal speed from policy acceleration, clips acceleration to `[-7.0, 3.5] m/s^2`, and uses a goal at `x=45.0` m in benchmark/dashboard runs. The configured emergency policy deceleration is `-6.5 m/s^2`; comfort/advisory values are `-2.5 m/s^2`.
- **Collision geometry:** oriented 2D bounding boxes with exact SAT collision checks and a conservative separation-distance calculation. Near-collision is separation below 1.5 m; the configured physical collision-distance threshold is 0.3 m. TTC warning/critical thresholds are 3.5/2.0 seconds in the risk estimator.
- **Policies:** reactive baseline, point-prediction-aware policy with constant-velocity fallback, uncertainty-aware policy, and validation-calibrated uncertainty policy. The main evaluator labels both CV and GRU point-prediction policies, but the evaluator constructs the same `PredictionAwarePolicy` without loading a GRU forecaster for that closed-loop benchmark; the saved aggregate results therefore do not establish a behavioral difference between CV and GRU.
- **Seeds and episode counts:** data generation uses default seed 42, batch base seed 1000, and 40 episodes per four scenarios. The main policy benchmark uses 10 episodes per scenario, 40 matched episodes total, and base seed 7000. The ablation uses six episodes with seeds 9001-9006. The three-scenario artifact records validation seeds 1101/1102 and final-test seeds 1201/1202/1203, 18 validation episodes plus 27 final-test episodes, 45 total. No confidence intervals are saved.
- **Action space:** scalar longitudinal acceleration, clipped by the simulator; action labels include cruising, advisory/proactive braking, uncertainty caution, and emergency braking.

### Website features

The active Streamlit dashboard provides:

- **Overview:** research flow and a preview simulation.
- **Scenario Lab:** scenario selection, policy selection, speed and scenario-specific controls, reproducible seed input, run button, KPI cards, recorded playback, frame scrubbing, actor selection, deterministic explanation text, technical telemetry, recorded events, speed/separation Plotly curves, and download of the recorded run.
- **Compare Methods:** matched one-episode comparison of reactive, point-prediction, and calibrated uncertainty policies, showing collision, near-collision, minimum separation, emergency actions, and elapsed time.
- **Data Explorer:** saved Indian-road audit JSON and availability of processed split files.
- **Results and Failure Cases:** saved forecasting, simulation, ablation, failure-case, three-scenario, and uncertainty-calibration outputs displayed as tables.
- **3D visualization:** local Three.js scene with road environment, actor meshes, history and forecast paths, replay ground truth, uncertainty overlays, follow/overhead/free-orbit/reset cameras, play/pause, reset, timeline scrubbing, playback speed, overlay toggles, and live speed/action/separation telemetry.
- **Limitations displayed:** the UI labels the experience as a synthetic research simulation and not a real-world vehicle controller. The repository limitations document states that the simulator is not deployment-ready.

### Generative AI

No generative-AI component is present in the repository. There is no LLM, text/image generation pipeline, prompt integration, or generative model. The GRU models are predictive trajectory models, not generative-AI systems.

## 2. Project Results

### Forecasting benchmark

Source: [outputs/metrics/forecasting_benchmark.json](outputs/metrics/forecasting_benchmark.json). The evaluator describes this as an unseen test-split evaluation. The saved processed manifest has 180 test windows. Metrics are mean errors in SI metres; no confidence intervals are provided.

| Model | ADE (m) | FDE (m) | Additional saved metrics |
| --- | ---: | ---: | --- |
| Constant velocity | 0.2184 | 0.5083 | None |
| Deterministic GRU | 0.3734 | 0.7468 | None |
| Uncertainty GRU mean | 6.7114 | 16.2240 | NLL 1.3105; 95% coverage 96.87%; 95% mean ellipse area 30.558 m2; 68% coverage 87.67%; 68% mean area 11.623 m2 |

The breakdown contains scenario/actor sample counts and standard deviations. The artifact does not contain confidence intervals. The uncertainty model's ADE/FDE are calculated from its predicted means, not from sampled trajectories.

### Training records

Source: [outputs/metrics/training_records.json](outputs/metrics/training_records.json). Both saved models trained for 25 epochs. Best validation values were deterministic GRU MSE 0.1640026 and uncertainty GRU NLL 1.3241184. The artifact contains histories but no training seed, confidence intervals, or hardware/runtime record.

### Main closed-loop policy benchmark

Source: [outputs/metrics/simulation_benchmark.json](outputs/metrics/simulation_benchmark.json). This is 40 matched episodes: 10 per each of the four main scenarios. Thresholds are from source configuration; no per-episode confidence intervals are saved.

| Policy | Collisions | Near-collision | Mean minimum separation | Emergency actions/episode | Completion | Mean elapsed time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reactive baseline | 0 / 40 | 75.0% | 1.198 m | 9.32 | 25.0% | 5.72 s |
| Prediction-aware CV | 0 / 40 | 37.5% | 3.323 m | 1.35 | 25.0% | 5.65 s |
| Prediction-aware GRU label | 0 / 40 | 37.5% | 3.323 m | 1.35 | 25.0% | 5.65 s |
| Uncertainty-aware | 0 / 40 | 35.0% | 3.937 m | 1.02 | 25.0% | 5.65 s |

Collision rate and collisions per 1,000 are 0.0 for all four rows. The saved file also reports unnecessary-intervention rates of 45.0% reactive, 0.0% for both point-prediction rows, and 0.0% uncertainty-aware. These are simulation outcomes only and do not establish real-world safety.

### Three-scenario matched evaluation

Source: [outputs/metrics/three_scenario_evaluation.json](outputs/metrics/three_scenario_evaluation.json). The artifact records no threshold tuning, validation seeds 1101/1102, final-test seeds 1201/1202/1203, and 45 total episodes. The final-test portion has three episodes per policy per scenario.

Final-test results:

| Scenario | Policy | Collisions | Near-collision | Mean separation | Emergency actions/episode | Completion | Mean time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pedestrian crossing | Reactive | 0/3 | 0.0% | 2.971 m | 11.33 | 0.0% | 6.0 s |
| Pedestrian crossing | Prediction-aware | 0/3 | 0.0% | 1.955 m | 3.33 | 0.0% | 6.0 s |
| Pedestrian crossing | Calibrated uncertainty | 0/3 | 0.0% | 3.358 m | 0.00 | 0.0% | 6.0 s |
| Autorickshaw cut-in | Reactive | 0/3 | 0.0% | 3.352 m | 0.00 | 100.0% | 5.9 s |
| Autorickshaw cut-in | Prediction-aware | 0/3 | 0.0% | 3.337 m | 0.00 | 100.0% | 5.8 s |
| Autorickshaw cut-in | Calibrated uncertainty | 0/3 | 0.0% | 2.787 m | 0.00 | 100.0% | 5.7 s |
| Animal entry | Reactive | 3/3 | 100.0% | 0.000 m | 9.00 | 0.0% | 3.0 s |
| Animal entry | Prediction-aware | 0/3 | 0.0% | 1.627 m | 6.00 | 0.0% | 6.0 s |
| Animal entry | Calibrated uncertainty | 0/3 | 0.0% | 3.713 m | 0.00 | 0.0% | 6.0 s |

The validation portion is also saved. It has six episodes per scenario across the two validation seeds, and the same artifact reports reactive collisions in both animal-entry validation episodes, with no collisions for the two predictive policies in that scenario. No confidence intervals are provided.

### Uncertainty calibration

Source: [outputs/metrics/uncertainty_calibration.json](outputs/metrics/uncertainty_calibration.json). Calibration uses 180 validation samples, a 0.90 quantile, 0.1 second horizons through 3.0 seconds, and a constant-velocity residual envelope. Mean marginal coverage is 90.0185%, minimum marginal coverage is 90.0%, final-horizon coverage is 90.0%, and mean ellipse area is 2.9136 m2. The file explicitly states that this is not joint trajectory coverage.

### Ablation and verification outputs

- [outputs/metrics/ablation_study.json](outputs/metrics/ablation_study.json) contains six-episode ablations for uncertainty quantiles 70%, 90%, 99% and proactive thresholds 1.5 m, 2.5 m, 4.0 m. All saved ablations report zero collisions, 33.33% completion, zero unnecessary interventions, and mean elapsed times from 5.48 to 5.52 s; mean minimum separation ranges from 4.186 to 5.722 m.
- [outputs/metrics/verification_pedestrian_autorickshaw.json](outputs/metrics/verification_pedestrian_autorickshaw.json) contains seed-42 hand-selected verification cases and matched policy examples. These are individual examples, not aggregate benchmark evidence.
- [outputs/metrics/failure_cases.json](outputs/metrics/failure_cases.json) contains episode-level entries for the main benchmark's near-collisions and unnecessary interventions. It does not prove deployment failure rates.

## 3. Evidence Checklist

### Tests executed

Executed during this inspection:

```text
python -m pytest tests/ -q
37 passed in 6.32s
```

The README says 22 tests, but the current repository run collected and passed 37 tests. No test failure was observed in this run. No coverage report was generated.

### Reproduction commands

The repository documents these commands; they are commands to run, not evidence that each was rerun during this inspection:

```bash
pip install -r requirements.txt
python scripts/run_dataset_audit.py
python scripts/generate_synthetic_data.py
python scripts/train_models.py
python scripts/run_evaluation.py
python scripts/generate_figures.py
python -m pytest tests/
streamlit run app.py
```

Additional saved-evaluation commands documented by the README are:

```bash
python scripts/calibrate_uncertainty.py
python scripts/run_three_scenario_evaluation.py --validation-seeds 1101 1102 --test-seeds 1201 1202 1203
```

Training is not rerun as part of this evidence inspection.

### Screenshots

No screenshot files are present in the repository. The dashboard can generate a live 3D view when launched, but that is not saved screenshot evidence.

### Graphs and plots

- [outputs/figures/trajectory_prediction_uncertainty.png](outputs/figures/trajectory_prediction_uncertainty.png): saved trajectory plot for a seed-42 pedestrian crossing, showing 2.0 s observed history, 3.0 s ground-truth future, constant-velocity forecast, and illustrative expanding 68%/95% ellipses generated by the figure script.
- [outputs/figures/policy_comparison_dynamics.png](outputs/figures/policy_comparison_dynamics.png): saved motorcycle cut-in dynamics plot for seed 1005, comparing speed profiles and separation curves for reactive, prediction-aware, and uncertainty-aware policies, with a 1.5 m near-collision line.

### Missing or incomplete evidence

- No confidence intervals, statistical significance tests, or uncertainty estimates for aggregate policy means are saved.
- No saved execution logs identify the exact historical runtime environment, Python patch version, hardware, or command output for the artifact-producing runs.
- The raw Indian Road Dataset is not included; its audit is documentary and the project does not establish real-world metric trajectory performance.
- No screenshots or browser acceptance results are saved.
- The main closed-loop evaluator labels a GRU policy but does not pass a trained GRU forecaster to `PredictionAwarePolicy`; its aggregate CV and GRU rows are identical and should not be described as a demonstrated GRU-vs-CV control comparison.
- LSTM and a learned risk classifier are absent.
- The saved plots are static generated figures, not evidence of live dashboard rendering.

## 4. Report-ready Extract

### Concise implementation summary

This repository implements a reproducible synthetic, metric 2D traffic simulator with six actor types, parameterized road-user scenarios, 10 Hz state updates, OBB/SAT collision geometry, TTC and clearance-based risk estimation, reactive and prediction-aware policies, a constant-velocity baseline, deterministic GRU forecasting, and bivariate-Gaussian uncertainty forecasting. A Streamlit dashboard embeds a procedural Three.js playback view and exposes scenario controls, policy comparisons, telemetry, explanations, saved metrics, and Plotly curves. The Indian Road Dataset is audited for taxonomy and data limitations, not used as metric trajectory ground truth.

### Concise results summary

On the saved 180-window forecasting test artifact, constant velocity records ADE/FDE of 0.2184/0.5083 m, deterministic GRU records 0.3734/0.7468 m, and the uncertainty GRU mean records 6.7114/16.2240 m with NLL 1.3105 and 96.87% empirical 95% ellipse coverage. On the saved 40-episode main policy benchmark, all policies record zero collisions; uncertainty-aware has the lowest near-collision rate (35.0%), highest mean minimum separation (3.937 m), and lowest mean emergency actions (1.02), while all policies have 25.0% task completion. In the separate final-test animal-entry scenario, reactive control records 3/3 collisions, while prediction-aware and calibrated uncertainty record 0/3; this is a small synthetic scenario result, not a general safety conclusion.

### Limitations

The results are synthetic and scenario-specific. Surrounding actors follow scripted trajectories and do not react to ego braking. The world is planar, dynamics are simplified, and the Indian-road imagery audit has sparse 1 Hz pseudo-labeled pixel tracks without the calibration needed for metric ground truth. A zero-collision result in any finite synthetic sample does not establish zero operational risk. The uncertainty calibration provides marginal per-horizon coverage rather than a joint trajectory guarantee. Aggregate confidence intervals and real-world validation are unavailable.

### Claims that must not be made

- Do not claim real-world autonomous-driving safety, deployment readiness, or collision-free operation.
- Do not claim the Indian Road Dataset supplied metric trajectory ground truth or that its raw videos were trained on.
- Do not claim an LSTM or a learned risk classifier exists.
- Do not claim generative AI is part of the system.
- Do not claim the main benchmark proves GRU policy superiority over CV; the saved CV and GRU policy rows are identical and the evaluator does not provide a GRU forecaster to that policy.
- Do not report confidence intervals, statistical significance, or broad population-level conclusions from the saved aggregate means.
- Do not present dashboard plots or the two static PNGs as screenshots or as proof of live browser rendering.