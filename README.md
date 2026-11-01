# Uncertainty-Aware Road-User Trajectory Prediction & Collision-Risk Analysis
### Indian-Road Scene Analysis and Synthetic Traffic Safety Evaluation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 📌 Research Overview

This software research prototype investigates whether **motion trajectory prediction** and **uncertainty-aware risk estimation** improve simulated collision-avoidance decisions compared with a purely **reactive safety baseline**.

### Key Research Questions:
1. **Movement History:** Does a 2.0-second observation history enable accurate forecasting of multi-agent positions over a 3.0-second horizon?
2. **Predictive Risk Assessment:** Does rolling forward predicted trajectories detect imminent collision and near-collision hazards earlier than reactive proximity rules?
3. **Uncertainty-Aware Safety Trade-Off:** Does propagating predictive covariance boundaries (Bivariate Gaussian NLL) improve the safety-efficiency trade-off by reducing late emergency braking while avoiding phantom emergency stops?

> **Important Boundary Disclaimers:**
> - This project is a **research simulation prototype**, not a real-vehicle controller.
> - "Indian-road-inspired" describes our choice of synthetic scenario archetypes (lane filtering, motorcycle cut-ins, pedestrian hesitation, and mixed-traffic intersections), not proof of real-world deployment readiness.
> - We strictly avoid fabricating false metric conversions from uncalibrated 2D monocular camera bounding boxes.

---

## 🔬 Experimental Findings & Benchmark Results

### 1. Trajectory Forecasting Benchmarks (Test Split, SI Metres)

Evaluated on 180 independent test trajectory windows across all 4 scenarios:

| Model Architecture | Output Representation | Overall ADE (m) | Overall FDE (m) | 95% Coverage | Mean Area (m²) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Constant Velocity Baseline** | Deterministic $(\Delta x, \Delta y)$ | **0.218 m** | **0.508 m** | — | — |
| **Deterministic TrajectoryGRU** | Neural Seq2Seq $(\Delta x, \Delta y)$ | **0.373 m** | **0.747 m** | — | — |
| **Uncertainty TrajectoryGRU** | Bivariate Gaussian $(\vec{\mu}, \vec{\sigma}, \rho)$ | **6.711 m** | **16.224 m** | **96.87%** | **30.6 m²** |

*Note: The Constant Velocity baseline provides a powerful linear benchmark on short horizons, while the Uncertainty GRU provides well-calibrated conservative coverage ($96.87\%$ empirical coverage at the $95\%$ nominal level).*

### 2. Closed-Loop Simulation Safety & Efficiency Benchmark

Evaluated across 40 matched test episodes under identical vehicle dynamics and action limits ($a \in [-6.5, 3.5]\text{ m/s}^2$):

| Safety Control Policy | Collisions | Collisions / 1,000 | Near-Collision Rate (<1.5m) | Mean Min Separation | Mean Emergency Actions / Ep | Mean Elapsed Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reactive Baseline** | 0 | 0.0 | 75.0% | 1.20 m | **9.32** | 5.72 s |
| **Prediction-Aware Policy (CV)** | 0 | 0.0 | 37.5% | 3.32 m | **1.35** | 5.65 s |
| **Prediction-Aware Policy (GRU)**| 0 | 0.0 | 37.5% | 3.32 m | **1.35** | 5.65 s |
| **Uncertainty-Aware Policy (90%)**| 0 | 0.0 | **35.0%** | **3.94 m** | **1.02** | 5.65 s |

**Key Finding:** 
The **Uncertainty-Aware Policy** reduces near-collision encounters from **75.0% down to 35.0%**, expands minimum clearance from **1.20m to 3.94m**, and cuts violent emergency braking interventions by **89%** (from 9.32 down to 1.02 actions/episode) without any penalty in travel time.

---

## 🗂️ Project Directory Structure

```text
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Python dependencies
├── README.md                   # Technical documentation and reproduction guide
├── configs/
│   └── default_config.yaml     # Master simulation & model configuration
├── src/
│   ├── data/
│   │   ├── dataset_audit.py    # Hugging Face Indian Road Dataset auditor
│   │   ├── synthetic_generator.py # Parameterized multi-agent scenario generator
│   │   └── data_pipeline.py    # Validation, zero-leakage splits, windowing
│   ├── models/
│   │   ├── constant_velocity.py# Constant-velocity physics baseline
│   │   ├── recurrent_forecaster.py # PyTorch GRU Seq2Seq model
│   │   └── uncertainty.py      # Bivariate Gaussian NLL Uncertainty GRU
│   ├── risk/
│   │   └── geometric_risk.py   # OBB SAT collision, metric TTC & risk scores
│   ├── simulation/
│   │   ├── geometry.py         # 2D OBB math, Separating Axis Theorem
│   │   ├── policies.py         # Reactive, Prediction-Aware, Uncertainty-Aware
│   │   └── simulator.py        # 2D closed-loop traffic simulation engine
│   └── evaluation/
│       ├── metrics.py          # ADE, FDE, NLL, Coverage, Simulation metrics
│       └── evaluator.py        # Benchmark runner & ablation coordinator
├── scripts/
│   ├── run_dataset_audit.py    # Executes dataset audit
│   ├── generate_synthetic_data.py # Generates dataset & processes windows
│   ├── train_models.py         # Trains PyTorch GRU & Uncertainty models
│   ├── run_evaluation.py       # Executes all benchmarks & ablations
│   └── generate_figures.py     # Generates publication-quality static plots
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_constant_velocity.py
│   ├── test_risk_and_geometry.py
│   ├── test_simulation_policies.py
│   ├── test_metrics.py
│   └── test_smoke_e2e.py
├── data/
│   ├── raw/                    # Audit metadata & schemas
│   ├── synthetic/              # Generated SI metric traffic episodes
│   └── processed/              # Train/Val/Test sliding window datasets
├── outputs/
│   ├── checkpoints/            # Trained PyTorch model weights (.pt)
│   ├── metrics/                # Experiment records, JSONs, CSVs
│   └── figures/                # Static evaluation figures (.png)
└── docs/
    ├── methodology.md          # Mathematical formulation & policy equations
    ├── dataset_audit.md        # Complete Indian Road Dataset audit report
    └── limitations.md          # Assumptions, boundaries, and disclaimers
```

---

## 🚀 Quick Start & Reproduction Guide

### 1. Installation
Clone repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline from Scratch
Execute the reproduction sequence:
```bash
# 1. Run Indian Road Dataset Audit (Data Source A)
python scripts/run_dataset_audit.py

# 2. Generate Synthetic Trajectories & Perform Zero-Leakage Splits (Data Source C)
python scripts/generate_synthetic_data.py

# 3. Train Deterministic & Uncertainty-Aware Forecasters
python scripts/train_models.py

# 4. Run Comprehensive Benchmark Evaluation & Ablation Study
python scripts/run_evaluation.py

# 5. Generate Figures
python scripts/generate_figures.py
```

### 3. Run Automated Tests
Execute the complete test suite (22 unit & smoke tests):
```bash
python -m pytest tests/
```

### 4. Launch the Streamlit Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```

The dashboard starts in **Scenario Lab** and runs a scenario only after the
explicit **Run scenario** action. Playback, camera controls, overlays, and
scrubbing operate on the recorded renderer payload; they do not rerun the
simulator or change its outcomes. The supported presets are pedestrian
crossing, motorcycle cut-in, mixed-traffic intersection, and unregulated
overtaking. Speeds are entered in km/h, pedestrian speed in m/s, and timing
in seconds; simulator state remains SI metres and seconds at 10 Hz.

To run the focused dashboard contract tests after installing dependencies:
```bash
python -m pytest tests/test_dashboard_integration.py -q
```

To run the matched three-scenario evaluation for pedestrian crossing,
auto-rickshaw cut-in, and sudden animal entry:
```bash
python scripts/calibrate_uncertainty.py
python scripts/run_three_scenario_evaluation.py --validation-seeds 1101 1102 --test-seeds 1201 1202 1203
```

The command saves episode-level configurations, aggregate policy metrics, and
prediction failure cases to `outputs/metrics/three_scenario_evaluation.json`.
Calibration uses only `data/processed/val_windows.npz`. The uncertainty policy
uses a 90% per-horizon marginal quantile of constant-velocity residuals; this
is not a joint trajectory coverage guarantee. Evaluation uses the existing
reactive, point-prediction, and calibrated uncertainty policies without model
training or final-test threshold tuning.

---

## 📜 Dataset Licensing & Citations

- **Indian Road Dataset:** ThirdEye Labs (`thirdeyelabs/indian-road-dataset`), licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Synthetic Data & Code:** Available for educational and academic research evaluation.
