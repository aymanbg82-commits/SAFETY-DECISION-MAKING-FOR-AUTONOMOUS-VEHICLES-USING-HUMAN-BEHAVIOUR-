"""Interactive 3D Traffic Research Dashboard.

The Python simulator remains the source of truth. Streamlit stores completed
runs in session state and the browser only replays the resulting payload.
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.simulation.policies import (
    CalibratedUncertaintyPolicy,
    PredictionAwarePolicy,
    ReactivePolicy,
    UncertaintyAwarePolicy,
)
from src.simulation.renderer_contract import build_renderer_payload
from src.simulation.simulator import TrafficSimulator
from src.visualization.explanations import generate_frame_explanation
from src.visualization.threejs_renderer import render_threejs_simulation_html


ROOT = Path(__file__).parent
SCENARIOS = {
    "pedestrian_crossing": "Pedestrian crossing",
    "autorickshaw_cut_in": "Auto-rickshaw cut-in",
    "animal_entry": "Sudden animal entry",
    "motorcycle_cut_in": "Motorcycle cut-in",
    "mixed_traffic_intersection": "Mixed-traffic intersection",
    "unregulated_overtaking": "Unregulated overtaking",
}
POLICIES = {
    "Reactive baseline": "reactive",
    "Point-prediction-aware (CV)": "prediction",
    "Calibrated uncertainty-aware (90% CV residual envelope)": "calibrated_uncertainty",
}


st.set_page_config(
    page_title="Traffic Research Dashboard",
    page_icon="TR",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .hero { padding: 0.5rem 0 1rem; border-bottom: 1px solid #d8dee8; margin-bottom: 1rem; }
    .eyebrow { color: #b45309; font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .disclaimer { background: #fff7ed; border: 1px solid #fdba74; color: #9a3412; padding: .6rem .8rem; border-radius: 6px; font-weight: 600; }
    .panel { background: #ffffff; border: 1px solid #d8dee8; border-radius: 7px; padding: 1rem; }
    .muted { color: #64748b; }
    .flow { color: #0f766e; font-weight: 700; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_json(relative_path: str) -> Optional[Dict[str, Any]]:
    path = ROOT / relative_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate only values supported by SyntheticTrajectoryGenerator."""
    if config["scenario"] not in SCENARIOS:
        return False, "Unsupported scenario preset."
    if not 5.0 <= config["speed_kmh"] <= 50.0:
        return False, "Initial speed must be between 5 and 50 km/h."
    if not 1 <= config["seed"] <= 99999:
        return False, "Random seed must be between 1 and 99,999."
    if not 0.5 <= config["ped_speed"] <= 2.5:
        return False, "Pedestrian speed must be between 0.5 and 2.5 m/s."
    if not 5.0 <= config.get("ped_start_distance", 35.0) <= 60.0:
        return False, "Pedestrian starting distance must be between 5 and 60 m."
    if not 0.0 <= config.get("crossing_time", 0.0) <= 4.0:
        return False, "Pedestrian crossing onset must be between 0 and 4 s."
    if not 5 <= config["cut_in_step"] <= 45:
        return False, "Motorcycle cut-in timing must be between 0.5 and 4.5 seconds."
    if config["scenario"] == "autorickshaw_cut_in":
        if not 5.0 <= config.get("cut_in_gap", 10.0) <= 30.0:
            return False, "Auto-rickshaw initial gap must be between 5 and 30 m."
        if not 5.0 <= config.get("cut_in_speed_kmh", 30.0) <= 50.0:
            return False, "Auto-rickshaw speed must be between 5 and 50 km/h."
        if not 0.5 <= config.get("cut_in_lateral_speed", 1.2) <= 3.0:
            return False, "Auto-rickshaw lateral speed must be between 0.5 and 3 m/s."
        if not 1.0 <= abs(config.get("cut_in_lateral_offset", 2.5)) <= 4.0:
            return False, "Auto-rickshaw lateral offset must be between 1 and 4 m."
    if config["scenario"] == "animal_entry":
        if not 5.0 <= config.get("animal_start_distance", 28.0) <= 60.0:
            return False, "Animal starting distance must be between 5 and 60 m."
        if not 0.0 <= config.get("animal_entry_time", 1.5) <= 4.5:
            return False, "Animal entry time must be between 0 and 4.5 s."
        if config.get("animal_direction", -1.0) not in (-1.0, 1.0):
            return False, "Animal movement direction must be -1 or 1."
        if not 0.5 <= config.get("animal_speed", 2.0) <= 6.0:
            return False, "Animal speed must be between 0.5 and 6 m/s."
        if not 0.3 <= config.get("animal_length", 1.2) <= 3.0 or not 0.2 <= config.get("animal_width", 0.7) <= 2.0:
            return False, "Animal dimensions are outside the supported synthetic range."
    return True, ""


def _make_policy(policy_key: str, target_speed: float):
    if policy_key == "reactive":
        return ReactivePolicy(target_speed=target_speed)
    if policy_key == "prediction":
        return PredictionAwarePolicy(target_speed=target_speed)
    if policy_key == "calibrated_uncertainty":
        profile = _read_json("outputs/metrics/uncertainty_calibration.json")
        if profile is None:
            raise ValueError("Calibrated uncertainty profile is unavailable; run scripts/calibrate_uncertainty.py")
        return CalibratedUncertaintyPolicy(profile, target_speed=target_speed)
    raise ValueError(f"Unsupported policy: {policy_key}")


@st.cache_data(show_spinner=False)
def run_config(config_json: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config = json.loads(config_json)
    speed_ms = config["speed_kmh"] / 3.6
    generator = SyntheticTrajectoryGenerator(time_step=0.1)
    episode = generator.generate_episode(
        config["scenario"],
        seed=config["seed"],
        custom_params={
            "ego_speed": speed_ms,
            "ped_speed": config["ped_speed"],
            "ped_start_distance": config.get("ped_start_distance", 35.0),
            "crossing_time": config.get("crossing_time", 0.0),
            "cut_in_step": config["cut_in_step"],
            "cut_in_gap": config.get("cut_in_gap", 10.0),
            "cut_in_speed": config.get("cut_in_speed_kmh", 30.0) / 3.6,
            "lateral_speed": config.get("cut_in_lateral_speed", 1.2),
            "lateral_offset": config.get("cut_in_lateral_offset", 2.5),
            "animal_start_distance": config.get("animal_start_distance", 28.0),
            "animal_entry_time": config.get("animal_entry_time", 1.5),
            "animal_direction": config.get("animal_direction", -1.0),
            "animal_speed": config.get("animal_speed", 2.0),
            "animal_length": config.get("animal_length", 1.2),
            "animal_width": config.get("animal_width", 0.7),
            "has_hesitation": config["hesitation"],
        },
    )
    policy = _make_policy(config["policy"], speed_ms)
    result = TrafficSimulator(dt=0.1, goal_x=45.0).run_episode(episode, policy)
    return result, build_renderer_payload(result, policy, dt=0.1), episode.parameters


def _current_config() -> Dict[str, Any]:
    scenario_label = st.selectbox("Scenario preset", list(SCENARIOS), format_func=SCENARIOS.get)
    speed_kmh = st.slider("Initial speed", 5.0, 50.0, 36.0, 1.0, format="%.0f km/h")
    ped_speed = 1.2
    ped_start_distance = 35.0
    crossing_time = 0.0
    cut_in_seconds = 1.5
    cut_in_gap = 10.0
    cut_in_speed_kmh = 30.0
    cut_in_lateral_speed = 1.2
    cut_in_lateral_offset = 2.5
    animal_start_distance = 28.0
    animal_entry_time = 1.5
    animal_direction = -1.0
    animal_speed = 2.0
    animal_length = 1.2
    animal_width = 0.7
    if scenario_label == "pedestrian_crossing":
        ped_speed = st.slider("Pedestrian walking speed", 0.5, 2.5, 1.2, 0.1, format="%.1f m/s")
        ped_start_distance = st.slider("Pedestrian starting distance", 5.0, 60.0, 35.0, 1.0, format="%.0f m")
        crossing_time = st.slider("Pedestrian crossing onset", 0.0, 4.0, 0.0, 0.1, format="%.1f s")
    elif scenario_label == "autorickshaw_cut_in":
        cut_in_gap = st.slider("Auto-rickshaw initial gap", 5.0, 30.0, 10.0, 1.0, format="%.0f m")
        cut_in_speed_kmh = st.slider("Auto-rickshaw speed", 5.0, 50.0, 30.0, 1.0, format="%.0f km/h")
        cut_in_lateral_speed = st.slider("Lateral movement speed", 0.5, 3.0, 1.2, 0.1, format="%.1f m/s")
        cut_in_lateral_offset = st.slider("Starting lateral offset", -4.0, 4.0, 2.5, 0.5, format="%.1f m")
        cut_in_seconds = st.slider("Cut-in timing", 0.5, 4.5, 1.5, 0.1, format="%.1f s")
    elif scenario_label == "animal_entry":
        animal_start_distance = st.slider("Animal starting distance", 5.0, 60.0, 28.0, 1.0, format="%.0f m")
        animal_entry_time = st.slider("Animal entry time", 0.0, 4.5, 1.5, 0.1, format="%.1f s")
        animal_direction = -1.0 if st.selectbox("Animal movement direction", ["Right to left", "Left to right"]) == "Right to left" else 1.0
        animal_speed = st.slider("Animal speed", 0.5, 6.0, 2.0, 0.1, format="%.1f m/s")
        animal_length = st.slider("Animal length", 0.3, 3.0, 1.2, 0.1, format="%.1f m")
        animal_width = st.slider("Animal width", 0.2, 2.0, 0.7, 0.1, format="%.1f m")
    else:
        cut_in_seconds = st.slider("Motorcycle cut-in timing", 0.5, 4.5, 1.5, 0.1, format="%.1f s")
    hesitation = st.checkbox("Enable pedestrian hesitation", value=True)
    policy_label = st.selectbox("Forecasting and safety policy", list(POLICIES))
    seed = st.number_input("Random seed", 1, 99999, 1005, 1)
    return {
        "scenario": scenario_label,
        "speed_kmh": float(speed_kmh),
        "ped_speed": float(ped_speed),
        "ped_start_distance": float(ped_start_distance),
        "crossing_time": float(crossing_time),
        "cut_in_gap": float(cut_in_gap),
        "cut_in_speed_kmh": float(cut_in_speed_kmh),
        "cut_in_lateral_speed": float(cut_in_lateral_speed),
        "cut_in_lateral_offset": float(cut_in_lateral_offset),
        "animal_start_distance": float(animal_start_distance),
        "animal_entry_time": float(animal_entry_time),
        "animal_direction": float(animal_direction),
        "animal_speed": float(animal_speed),
        "animal_length": float(animal_length),
        "animal_width": float(animal_width),
        "cut_in_step": int(round(cut_in_seconds / 0.1)),
        "hesitation": bool(hesitation),
        "policy": POLICIES[policy_label],
        "policy_label": policy_label,
        "seed": int(seed),
    }


def _metric_summary(result: Dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Collision", "Yes" if result["collision"] else "No")
    cols[1].metric("Near-collision", "Yes" if result["near_collision"] else "No")
    cols[2].metric("Minimum separation", f"{result['min_separation']:.2f} m")
    cols[3].metric("Emergency actions", result["emergency_braking_count"])
    cols[4].metric("Completion", "Completed" if result["task_completed"] else "Timeout / halted")


def _render_lab(config: Dict[str, Any], result: Dict[str, Any], payload: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    _metric_summary(result)
    st.caption("The viewport replays recorded simulator frames. Interpolation affects display smoothness only.")
    components.html(render_threejs_simulation_html(payload, height=580), height=600, scrolling=False)

    frames = payload.get("frames", [])
    if not frames:
        st.warning("The simulator returned no frames for this configuration.")
        return
    st.subheader("Actor inspection and explanation")
    left, right = st.columns([1, 2])
    with left:
        frame_index = st.slider("Inspection timestamp", 0, len(frames) - 1, 0, format="frame %d")
        actor_options = [a["id"] for a in payload.get("actors", [])]
        selected_actor = st.selectbox("Selected actor", actor_options)
        st.write({"actor": selected_actor, "time_s": frames[frame_index]["timestamp"]})
    frame = frames[frame_index]
    explanation = generate_frame_explanation(
        frame,
        {actor["id"]: actor for actor in payload.get("actors", [])},
        result["policy_name"],
        config["scenario"],
    )
    with right:
        st.markdown(f"**Situation:** {explanation['situation']}")
        st.markdown(f"**Prediction:** {explanation['prediction']}")
        st.markdown(f"**Risk factors:** {explanation['risk_detail']}")
        st.markdown(f"**Selected response:** {explanation['simulated_response']}")
        with st.expander("Technical values and assumptions"):
            st.json(explanation["technical_telemetry"])

    st.subheader("Recorded events")
    if payload.get("events"):
        st.dataframe(pd.DataFrame(payload["events"]), hide_index=True, width="stretch")
    else:
        st.info("No events were recorded for this episode.")
    logs = result["step_logs"]
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=[x["timestamp"] for x in logs], y=[x["ego_speed"] * 3.6 for x in logs], name="Speed (km/h)"))
    chart.add_trace(go.Scatter(x=[x["timestamp"] for x in logs], y=[x["min_separation"] for x in logs], name="Separation (m)"))
    chart.update_layout(height=320, xaxis_title="Simulation time (s)", yaxis_title="Measured value")
    st.plotly_chart(chart, width="stretch")
    st.download_button("Download recorded run", json.dumps({"config": config, "parameters": parameters, "result": result}, indent=2), "traffic_run.json", "application/json")


def _load_metrics_page() -> None:
    metrics = {
        "forecasting": _read_json("outputs/metrics/forecasting_benchmark.json"),
        "simulation": _read_json("outputs/metrics/simulation_benchmark.json"),
        "ablation": _read_json("outputs/metrics/ablation_study.json"),
        "failures": _read_json("outputs/metrics/failure_cases.json"),
        "three_scenario_evaluation": _read_json("outputs/metrics/three_scenario_evaluation.json"),
        "uncertainty_calibration": _read_json("outputs/metrics/uncertainty_calibration.json"),
    }
    for name, value in metrics.items():
        st.subheader(name.title())
        if value is None:
            st.info("Not evaluated yet or saved output is unavailable.")
        elif isinstance(value, list):
            st.dataframe(pd.DataFrame(value), hide_index=True, width="stretch")
        else:
            st.dataframe(pd.DataFrame(value).T, width="stretch")


def _run_comparison(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    outputs = {}
    for label, key in POLICIES.items():
        compare_config = dict(config, policy=key, policy_label=label)
        result, _, _ = run_config(json.dumps(compare_config, sort_keys=True))
        outputs[label] = result
    return outputs


st.markdown('<div class="hero"><div class="eyebrow">Synthetic research environment</div><h1>Traffic Research Dashboard</h1><div class="muted">Movement history → prediction → risk → simulated response</div></div>', unsafe_allow_html=True)
st.markdown('<div class="disclaimer">Synthetic research simulation — not a real-world driving safety system.</div>', unsafe_allow_html=True)

pages = ["Overview", "Scenario Lab", "Compare Methods", "Data Explorer", "Results and Failure Cases"]
requested_page = st.session_state.pop("nav_target", "Scenario Lab")
page = st.sidebar.radio("Section", pages, index=pages.index(requested_page), key="page")
st.sidebar.caption("Units: metres, seconds, m/s; simulator timestep: 0.1 s")

if page == "Overview":
    st.header("Research overview")
    st.write("This dashboard exposes the existing synthetic trajectory generator, closed-loop policies, geometric risk checks, and saved experiment records in one inspectable interface.")
    st.markdown('<p class="flow">Observed history &nbsp;→&nbsp; forecast &nbsp;→&nbsp; geometric risk &nbsp;→&nbsp; policy action</p>', unsafe_allow_html=True)
    st.info("A single episode is a demonstration. Aggregate benchmark files are separate evidence and should not be interpreted as proof of real-world safety.")
    if st.button("Open Scenario Lab", type="primary"):
        st.session_state["nav_target"] = "Scenario Lab"
        st.rerun()
    st.markdown("Dataset audit and methodology are available in `docs/dataset_audit.md` and `docs/methodology.md`.")
    st.subheader("Supported components")
    st.write("Scenarios: " + ", ".join(SCENARIOS.values()))
    st.write("Actor dimensions and movement come from the simulator's documented actor specifications; the website does not infer extra vehicle dynamics.")

elif page == "Scenario Lab":
    st.header("Scenario Lab")
    with st.sidebar:
        st.subheader("Run configuration")
    config = _current_config()
    valid, message = validate_config(config)
    if not valid:
        st.error(message)
    active_config = st.session_state.get("active_config")
    is_outdated = active_config is not None and active_config != config
    if is_outdated:
        st.warning("Configuration changed. Run the scenario to update the playback.")
    if st.button("Run scenario", type="primary", disabled=not valid):
        result, payload, parameters = run_config(json.dumps(config, sort_keys=True))
        st.session_state["active_config"] = config
        st.session_state["active_run"] = (result, payload, parameters)
        active_config = config
    if "active_run" not in st.session_state:
        st.info("Configure a supported scenario and select Run scenario to create a recorded playback.")
    else:
        result, payload, parameters = st.session_state["active_run"]
        _render_lab(active_config or config, result, payload, parameters)

elif page == "Compare Methods":
    st.header("Compare Methods")
    st.caption("Matched one-episode comparison: same scenario configuration, seed, initial conditions, and action limits.")
    config = _current_config()
    if st.button("Run matched comparison", type="primary"):
        st.session_state["comparison"] = _run_comparison(config)
        st.session_state["comparison_config"] = config
    comparison = st.session_state.get("comparison")
    if comparison:
        rows = []
        for label, result in comparison.items():
            rows.append({"method": label, "collision": result["collision"], "near_collision": result["near_collision"], "minimum_separation_m": result["min_separation"], "emergency_actions": result["emergency_braking_count"], "elapsed_s": result["elapsed_time"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Measured differences are simulation-specific outcomes, not driving advice or general safety claims.")
    else:
        st.info("Run the matched comparison to measure all available policies.")

elif page == "Data Explorer":
    st.header("Data Explorer")
    audit = _read_json("outputs/metrics/indian_road_dataset_audit.json")
    if audit is None:
        st.info("Indian-road audit output is unavailable. Run `python scripts/run_dataset_audit.py` to generate it.")
    else:
        st.subheader("Indian-road dataset audit")
        st.json(audit)
    st.subheader("Synthetic data")
    st.write("Synthetic episodes use SI metres, a 0.1 second sampling interval, explicit actor dimensions, and reproducible seeds.")
    for split in ("train_windows.npz", "val_windows.npz", "test_windows.npz"):
        path = ROOT / "data" / "processed" / split
        st.write(f"{split}: {'available' if path.exists() else 'unavailable'}")

else:
    st.header("Results and Failure Cases")
    _load_metrics_page()

# The legacy dashboard was retained below during migration. Stop execution
# here so only the current dashboard renders.
st.stop()
"""
Interactive 3D Traffic Research Dashboard:
Uncertainty-Aware Road-User Trajectory Prediction and Collision-Risk Analysis
(Indian-Road Scene Analysis and Synthetic Traffic Safety Evaluation).
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

from src.data.synthetic_generator import SyntheticTrajectoryGenerator
from src.simulation.simulator import TrafficSimulator
from src.simulation.policies import ReactivePolicy, PredictionAwarePolicy, UncertaintyAwarePolicy
from src.simulation.renderer_contract import build_renderer_payload
from src.visualization.threejs_renderer import render_threejs_simulation_html
from src.visualization.explanations import generate_frame_explanation


# -----------------------------------------------------------------------------
# Streamlit App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="3D Traffic Research Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme CSS (Light Theme: #F8FAFC, #FFFFFF, #0F172A, #2563EB)
st.markdown("""
<style>
    /* Global Fonts & Palette */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0F172A;
        background-color: #F8FAFC;
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 0.15rem;
    }
    .main-subtitle {
        font-size: 1.0rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 1.0rem;
    }
    .disclaimer-badge {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }
    .card-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.0rem;
    }
    .stat-label {
        font-size: 12px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
    }
    .explanation-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-top: 8px;
        font-size: 14px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_audit_data():
    audit_path = os.path.join("outputs", "metrics", "indian_road_dataset_audit.json")
    if os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_benchmark_metrics():
    res = {}
    f_path = os.path.join("outputs", "metrics", "forecasting_benchmark.json")
    s_path = os.path.join("outputs", "metrics", "simulation_benchmark.json")
    a_path = os.path.join("outputs", "metrics", "ablation_study.json")
    fail_path = os.path.join("outputs", "metrics", "failure_cases.json")

    if os.path.exists(f_path):
        with open(f_path, "r", encoding="utf-8") as f:
            res["forecasting"] = json.load(f)
    if os.path.exists(s_path):
        with open(s_path, "r", encoding="utf-8") as f:
            res["simulation"] = json.load(f)
    if os.path.exists(a_path):
        with open(a_path, "r", encoding="utf-8") as f:
            res["ablation"] = json.load(f)
    if os.path.exists(fail_path):
        with open(fail_path, "r", encoding="utf-8") as f:
            res["failures"] = json.load(f)
    return res


# -----------------------------------------------------------------------------
# Top Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Uncertainty-Aware Road-User Trajectory Prediction & Collision-Risk Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Indian-Road Scene Analysis & Synthetic Traffic Safety Evaluation — 3D Research Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="disclaimer-badge">⚠️ SYNTHETIC RESEARCH SIMULATION — NOT A REAL-WORLD VEHICLE CONTROLLER</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to page:",
    [
        "Overview",
        "Scenario Lab (3D View)",
        "Compare Methods",
        "Data Explorer",
        "Results & Failure Cases"
    ],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Research Prototype v1.0.0")
st.sidebar.caption("Units: SI Metres (m), Seconds (s), 10 Hz dt")


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if page_selection == "Overview":
    st.subheader("Research Objective & Methodology Flow")
    st.markdown("""
    This project investigates whether **motion trajectory forecasting** and **predictive uncertainty quantification**
    improve simulated safety decisions compared with a traditional **reactive collision-avoidance baseline**.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card-panel"><div class="stat-label">Step 1</div><div class="stat-value">Perception</div><p style="font-size:13px; color:#475569; margin-top:6px;">2.0s observation history of actor positions and velocities in SI metres.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card-panel"><div class="stat-label">Step 2</div><div class="stat-value">Forecasting</div><p style="font-size:13px; color:#475569; margin-top:6px;">3.0s future path projection via Constant Velocity or Bivariate Gaussian GRU.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card-panel"><div class="stat-label">Step 3</div><div class="stat-value">Risk Estimation</div><p style="font-size:13px; color:#475569; margin-top:6px;">Exact OBB collision geometry, minimum separation, and conservative uncertainty clearance.</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card-panel"><div class="stat-label">Step 4</div><div class="stat-value">Action</div><p style="font-size:13px; color:#475569; margin-top:6px;">Closed-loop deceleration commands (cruising, proactive braking, or emergency stopping).</p></div>', unsafe_allow_html=True)

    st.markdown("### 3D Preview: Indian-Road Inspired Pedestrian Crossing")
    st.caption("Lightweight preview of a pedestrian stepping onto an unseparated urban corridor with mid-street hesitation.")

    # Generate a lightweight preview episode
    gen_prev = SyntheticTrajectoryGenerator(time_step=0.1)
    ep_prev = gen_prev.generate_episode("pedestrian_crossing", seed=42)
    sim_prev = TrafficSimulator(dt=0.1, goal_x=40.0)
    res_prev = sim_prev.run_episode(ep_prev, UncertaintyAwarePolicy(target_speed=10.0))
    payload_prev = build_renderer_payload(res_prev, UncertaintyAwarePolicy(target_speed=10.0))

    html_prev = render_threejs_simulation_html(payload_prev, height=480)
    components.html(html_prev, height=500, scrolling=False)


# =============================================================================
# PAGE 2: SCENARIO LAB (3D VIEW)
# =============================================================================
elif page_selection == "Scenario Lab (3D View)":
    st.subheader("Scenario Lab — Interactive 3D Traffic Simulation")
    st.caption("Configure Indian-road scenario archetypes, execute the closed-loop simulator, and inspect 3D trajectories in real time.")

    # Two-Column Layout: Controls on Left, 3D Scene + Telemetry on Right
    left_col, right_col = st.columns([1, 2.8])

    with left_col:
        st.markdown("#### Scenario Configuration")
        scenario_type = st.selectbox(
            "Scenario Archetype",
            ["pedestrian_crossing", "motorcycle_cut_in", "mixed_traffic_intersection", "unregulated_overtaking"],
            format_func=lambda x: x.replace("_", " ").title(),
            index=1
        )
        policy_type = st.selectbox(
            "Safety Control Policy",
            ["Reactive Baseline", "Prediction-Aware (CV)", "Uncertainty-Aware (90% Envelope)"],
            index=2
        )
        sim_seed = st.number_input("Random Seed (Reproducible)", min_value=1, max_value=99999, value=1005, step=1)
        ego_speed_kmh = st.slider("Ego Target Speed (km/h)", min_value=20.0, max_value=50.0, value=36.0, step=2.0)

        with st.expander("Advanced Actor Parameters"):
            st.caption("Parameter ranges supported by simulator:")
            crossing_speed = st.slider("Obstacle Speed Modifier", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            hesitation_toggle = st.checkbox("Enable Hesitation (Pedestrian/Two-Wheeler)", value=True)

        run_btn = st.button("▶ Run Scenario in 3D", type="primary", width="stretch")

    # Execute simulation using Python simulator as single source of truth
    gen = SyntheticTrajectoryGenerator(time_step=0.1)
    ego_speed_ms = ego_speed_kmh / 3.6

    ep = gen.generate_episode(
        scenario_type=scenario_type,
        seed=int(sim_seed),
        custom_params={"ego_speed": ego_speed_ms, "has_hesitation": hesitation_toggle}
    )

    if policy_type == "Reactive Baseline":
        pol = ReactivePolicy(target_speed=ego_speed_ms)
    elif policy_type == "Prediction-Aware (CV)":
        pol = PredictionAwarePolicy(target_speed=ego_speed_ms)
    else:
        pol = UncertaintyAwarePolicy(target_speed=ego_speed_ms, conservative_quantile=0.90)

    sim = TrafficSimulator(dt=0.1, goal_x=45.0)
    sim_result = sim.run_episode(ep, pol)

    # Build 3D payload
    payload = build_renderer_payload(sim_result, pol, dt=0.1)

    with right_col:
        # Top KPI bar
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Collision", "COLLISION" if sim_result["collision"] else "SAFE", delta_color="inverse")
        kpi2.metric("Near-Collision (<1.5m)", "YES" if sim_result["near_collision"] else "NO", delta_color="inverse")
        kpi3.metric("Min Separation", f"{sim_result['min_separation']:.2f} m")
        kpi4.metric("Emergency Brakes", f"{sim_result['emergency_braking_count']}")
        kpi5.metric("Task Outcome", "COMPLETED" if sim_result["task_completed"] else "HALTED")

        # Render Three.js 3D Viewport
        html_code = render_threejs_simulation_html(payload, height=560)
        components.html(html_code, height=580, scrolling=False)

    # Bottom Inspection & Explanation Section
    st.markdown("---")
    st.markdown("### Risk & Action Explanation Engine")
    st.caption("Select any frame timestamp to inspect the deterministic reasoning behind the policy's action.")

    frames = payload["frames"]
    total_frames = len(frames)

    exp_col1, exp_col2 = st.columns([1.2, 2.8])
    with exp_col1:
        frame_slider = st.slider("Inspect Frame Step", 0, max(0, total_frames - 1), min(15, total_frames - 1))
        selected_frame = frames[frame_slider] if frames else {}

        actors_dict = {a["id"]: a for a in payload["actors"]}
        explanation = generate_frame_explanation(selected_frame, actors_dict, pol.__class__.__name__, scenario_type)

        st.markdown(f"**Timestamp:** `{explanation['timestamp']:.2f} s`")
        st.markdown(f"**Risk Level:** `{explanation['risk_category']}`")
        st.markdown(f"**Triggered Rule:** *{explanation['risk_rule']}*")

    with exp_col2:
        st.markdown(f"""
        <div class="explanation-box">
            <b>1. Situation:</b> {explanation['situation']}<br><br>
            <b>2. Trajectory Prediction:</b> {explanation['prediction']}<br><br>
            <b>3. Risk Factors:</b> {explanation['risk_detail']}<br><br>
            <b>4. Simulated Response:</b> {explanation['simulated_response']}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View Frame Technical Telemetry (SI Units)"):
            st.json(explanation["technical_telemetry"])

    # Bottom Interactive Plots
    st.markdown("---")
    st.markdown("### Episode Kinematics & Clearance Curves")
    logs = sim_result["step_logs"]
    times = [l["timestamp"] for l in logs]
    speeds = [l["ego_speed"] * 3.6 for l in logs]
    accels = [l["acceleration"] for l in logs]
    separations = [l["min_separation"] for l in logs]

    fig_kin = go.Figure()
    fig_kin.add_trace(go.Scatter(x=times, y=speeds, name="Ego Speed (km/h)", line=dict(color="#2563EB", width=2.5)))
    fig_kin.add_trace(go.Scatter(x=times, y=separations, name="Clearance (m)", line=dict(color="#10B981", width=2.0, dash="dash")))
    fig_kin.add_trace(go.Scatter(x=times, y=accels, name="Acceleration (m/s²)", line=dict(color="#F59E0B", width=1.5, dash="dot")))
    fig_kin.add_hline(y=1.5, line_color="#EF4444", line_dash="dot", annotation_text="Near-Collision Threshold (1.5m)")
    fig_kin.update_layout(
        title="Kinematics & Separation Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Metric Value",
        height=320,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_kin, width="stretch")


# =============================================================================
# PAGE 3: COMPARE METHODS
# =============================================================================
elif page_selection == "Compare Methods":
    st.subheader("Compare Safety Policies on Matched Scenarios")
    st.markdown("""
    Evaluate how **Reactive Baseline**, **Point-Prediction-Aware**, and **Uncertainty-Aware** policies
    handle the identical traffic encounter under identical vehicle dynamics and action limits.
    """)

    c_col1, c_col2 = st.columns([1, 3])
    with c_col1:
        comp_scenario = st.selectbox(
            "Select Benchmark Scenario",
            ["motorcycle_cut_in", "pedestrian_crossing", "mixed_traffic_intersection", "unregulated_overtaking"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="comp_scen_select"
        )
        comp_seed_val = st.number_input("Matched Seed", value=1005, min_value=1, key="comp_seed_val")
        comp_spd = st.slider("Target Speed (km/h)", 25.0, 45.0, 36.0, 1.0, key="comp_spd_val")

    gen_cmp = SyntheticTrajectoryGenerator(time_step=0.1)
    ep_cmp = gen_cmp.generate_episode(scenario_type=comp_scenario, seed=int(comp_seed_val), custom_params={"ego_speed": comp_spd / 3.6})

    sim_cmp = TrafficSimulator(dt=0.1, goal_x=45.0)

    res_react = sim_cmp.run_episode(ep_cmp, ReactivePolicy(target_speed=comp_spd / 3.6))
    res_pred = sim_cmp.run_episode(ep_cmp, PredictionAwarePolicy(target_speed=comp_spd / 3.6))
    res_unc = sim_cmp.run_episode(ep_cmp, UncertaintyAwarePolicy(target_speed=comp_spd / 3.6, conservative_quantile=0.90))

    with c_col2:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("### 🔴 Reactive Baseline")
            st.metric("Min Separation", f"{res_react['min_separation']:.2f} m")
            st.metric("Emergency Brakes", f"{res_react['emergency_braking_count']}")
            st.metric("Near-Collision (<1.5m)", "YES" if res_react['near_collision'] else "NO")
            st.metric("Elapsed Time", f"{res_react['elapsed_time']:.1f} s")

        with m2:
            st.markdown("### 🔵 Prediction-Aware (CV)")
            st.metric("Min Separation", f"{res_pred['min_separation']:.2f} m", delta=f"{res_pred['min_separation'] - res_react['min_separation']:+.2f} m")
            st.metric("Emergency Brakes", f"{res_pred['emergency_braking_count']}", delta=f"{res_pred['emergency_braking_count'] - res_react['emergency_braking_count']}")
            st.metric("Near-Collision (<1.5m)", "YES" if res_pred['near_collision'] else "NO")
            st.metric("Elapsed Time", f"{res_pred['elapsed_time']:.1f} s")

        with m3:
            st.markdown("### 🟢 Uncertainty-Aware (90%)")
            st.metric("Min Separation", f"{res_unc['min_separation']:.2f} m", delta=f"{res_unc['min_separation'] - res_react['min_separation']:+.2f} m")
            st.metric("Emergency Brakes", f"{res_unc['emergency_braking_count']}", delta=f"{res_unc['emergency_braking_count'] - res_react['emergency_braking_count']}")
            st.metric("Near-Collision (<1.5m)", "YES" if res_unc['near_collision'] else "NO")
            st.metric("Elapsed Time", f"{res_unc['elapsed_time']:.1f} s")

    # Speed Profile Comparison Chart
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=[l["timestamp"] for l in res_react["step_logs"]],
        y=[l["ego_speed"] * 3.6 for l in res_react["step_logs"]],
        name="Reactive Baseline", line=dict(color="#EF4444", width=2.5)
    ))
    fig_cmp.add_trace(go.Scatter(
        x=[l["timestamp"] for l in res_pred["step_logs"]],
        y=[l["ego_speed"] * 3.6 for l in res_pred["step_logs"]],
        name="Prediction-Aware Policy", line=dict(color="#3B82F6", width=2.5, dash="dash")
    ))
    fig_cmp.add_trace(go.Scatter(
        x=[l["timestamp"] for l in res_unc["step_logs"]],
        y=[l["ego_speed"] * 3.6 for l in res_unc["step_logs"]],
        name="Uncertainty-Aware Policy", line=dict(color="#10B981", width=3)
    ))
    fig_cmp.update_layout(
        title=f"Deceleration Profiles During {comp_scenario.replace('_', ' ').title()}",
        xaxis_title="Simulation Time (seconds)",
        yaxis_title="Ego Speed (km/h)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cmp, width="stretch")


# =============================================================================
# PAGE 4: DATA EXPLORER
# =============================================================================
elif page_selection == "Data Explorer":
    st.subheader("Data Explorer: Real Dataset Audit & Synthetic Benchmark")

    data_tabs = st.tabs(["🏛️ Hugging Face Indian Road Dataset", "🧪 Synthetic Multi-Agent Dataset"])

    with data_tabs[0]:
        st.markdown("### Data Source A: ThirdEye Labs Indian Road Driving Dataset")
        st.markdown("Official Hugging Face Hub: [`thirdeyelabs/indian-road-dataset`](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset)")

        audit = load_audit_data()
        if audit:
            m = audit["metadata"]
            feas = audit["feasibility_assessment"]

            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            col_a1.metric("Storage Volume", f"{m['total_storage_gb']} GB", "646 Tar Shards")
            col_a2.metric("Total Video Clips", f"{m['total_clips']:,}")
            col_a3.metric("Keyframe Detections", f"{m['object_detections']:,}")
            col_a4.metric("Sampling Rate", f"{m['keyframe_sampling_rate_hz']} Hz", "1 FPS via FFmpeg")

            st.markdown("#### Verified Object Class Taxonomy (12 Classes)")
            cls_cols = st.columns(6)
            for i, c in enumerate(m["classes"]):
                with cls_cols[i % 6]:
                    st.markdown(f"**`{c['name']}`**")
                    st.caption(c["description"])

            st.markdown("---")
            st.markdown("#### Technical Feasibility: Why Monocular Bounding Boxes Cannot Form Metric Trajectories")
            st.error(f"**1. Coordinate Frame:** {feas['coordinate_frame_gap']}")
            st.warning(f"**2. Temporal Sampling:** {feas['sampling_frequency_limitation']}")
            st.info(f"**3. Physical TTC Invalidity:** {feas['ttc_calculation_invalidity']}")
            st.success(f"**4. Project Strategy:** {feas['conclusion_and_role_in_project']}")

    with data_tabs[1]:
        st.markdown("### Data Source C: Synthetic Metric Trajectory Dataset")
        st.markdown("""
        All trajectory models and safety policies are evaluated in our reproducible synthetic 2D environment
        with known physical units (**SI Metres**, uniform **10 Hz sampling**, explicit bounding boxes).
        """)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total State Records", "24,000", "160 Episodes")
        col_s2.metric("Train Split", "840 Windows", "112 Episodes (70%)")
        col_s3.metric("Val Split", "180 Windows", "24 Episodes (15%)")
        col_s4.metric("Test Split", "180 Windows", "24 Episodes (15%)")

        st.caption("Zero-leakage split guarantee: All sliding windows from the same episode remain strictly within one split.")


# =============================================================================
# PAGE 5: RESULTS & FAILURE CASES
# =============================================================================
elif page_selection == "Results & Failure Cases":
    st.subheader("Official Experiment Records & Failure Case Inspector")

    metrics = load_benchmark_metrics()

    res_tab1, res_tab2, res_tab3 = st.tabs([
        "📊 Policy Simulation Benchmarks",
        "🎯 Trajectory Forecasting Metrics",
        "⚠️ Failure Case Inspector"
    ])

    with res_tab1:
        st.markdown("### Closed-Loop Safety & Efficiency Benchmark (Matched Episodes)")
        if "simulation" in metrics:
            sim_df = pd.DataFrame(metrics["simulation"]).T
            st.dataframe(sim_df, width="stretch")

        if "ablation" in metrics:
            st.markdown("### Component Ablation Study (Uncertainty Quantile & Proactive Thresholds)")
            abl_df = pd.DataFrame(metrics["ablation"]).T
            st.dataframe(abl_df, width="stretch")

    with res_tab2:
        st.markdown("### Trajectory Forecasting Benchmarks (Unseen Test Split)")
        if "forecasting" in metrics:
            fc = metrics["forecasting"]
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.markdown("#### Constant Velocity")
                st.metric("Overall ADE", f"{fc['constant_velocity']['overall_ade_meters']:.3f} m")
                st.metric("Overall FDE", f"{fc['constant_velocity']['overall_fde_meters']:.3f} m")
            with f_col2:
                st.markdown("#### Deterministic GRU")
                st.metric("Overall ADE", f"{fc['deterministic_gru']['overall_ade_meters']:.3f} m")
                st.metric("Overall FDE", f"{fc['deterministic_gru']['overall_fde_meters']:.3f} m")
            with f_col3:
                st.markdown("#### Probabilistic GRU (NLL)")
                st.metric("Test NLL", f"{fc['uncertainty_gru']['negative_log_likelihood']:.2f}")
                st.metric("95% Empirical Coverage", f"{fc['uncertainty_gru']['coverage_95_percent']}%", f"Area: {fc['uncertainty_gru']['mean_ellipse_area_95_m2']:.1f} m²")

    with res_tab3:
        st.markdown("### Recorded Near-Collision and Emergency Intervention Failure Cases")
        st.markdown("Inspecting cases where policies approached within $< 1.5\text{m}$ clearance or triggered emergency friction stops.")

        if "failures" in metrics:
            fails = metrics["failures"]
            st.caption(f"Total recorded critical events: {len(fails)}")
            st.dataframe(pd.DataFrame(fails), width="stretch")
