"""
Simulation-to-3D Renderer Integration Contract.
Converts closed-loop simulation runs, model forecasts, and risk assessments
into a standardized, structured JSON payload for browser-based Three.js rendering.
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np

from src.simulation.geometry import OrientedBoundingBox, compute_obb_separation_distance
from src.models.constant_velocity import ConstantVelocityForecaster
from src.simulation.policies import BasePolicy


# Color mappings for 3D meshes (Hex strings)
ACTOR_COLORS = {
    "car": "#2563EB",            # Deep Blue (Ego) or #475569 for other cars
    "motorcycle": "#EA580C",     # Orange
    "autorickshaw": "#CA8A04",   # Amber / Yellow
    "pedestrian": "#16A34A",     # Green
    "animal": "#A16207",         # Synthetic animal
    "truck": "#64748B",          # Slate
}

# Coordinate Mapping documentation:
# Simulator frame: 2D Cartesian plane (SI Metres)
#   X_sim: longitudinal along road (East / forward)
#   Y_sim: lateral across road lanes (North / left = positive, right = negative)
#   Z_sim: height above road surface (0 = road surface)
# Three.js 3D scene frame:
#   X_3d = X_sim
#   Y_3d = elevation above ground (Up)
#   Z_3d = -Y_sim (lateral lane offset)
# Heading rotation in Three.js:
#   mesh.rotation.y = heading_sim (radians)


def build_renderer_payload(
    episode_result: Dict[str, Any],
    policy: BasePolicy,
    forecaster_model: Optional[Any] = None,
    dt: float = 0.1
) -> Dict[str, Any]:
    """
    Build a complete, self-contained playback payload for the 3D renderer.

    Args:
        episode_result: Output dictionary from TrafficSimulator.run_episode().
        policy: The policy instance executed.
        forecaster_model: Optional trained model for generating frame predictions.
        dt: Time step duration (seconds).

    Returns:
        Dict conforming strictly to the 3D renderer integration contract.
    """
    step_logs = episode_result["step_logs"]
    num_steps = len(step_logs)

    # 1. Identify distinct actors and their physical specifications
    actors_meta = {}
    # Ego vehicle is always present
    actors_meta["ego"] = {
        "id": "ego",
        "name": "Ego Vehicle",
        "type": "car",
        "length": 4.5,
        "width": 2.0,
        "height": 1.45,
        "color": "#2563EB",
        "is_ego": True
    }

    for log in step_logs:
        for a in log["surrounding_actors"]:
            aid = a["actor_id"]
            if aid not in actors_meta:
                atype = a.get("actor_type", "car")
                actors_meta[aid] = {
                    "id": aid,
                    "name": f"{atype.replace('_', ' ').title()} ({aid})",
                    "type": atype,
                    "length": float(a.get("length", 3.0)),
                    "width": float(a.get("width", 1.5)),
                    "height": 1.0 if atype == "animal" else (1.75 if atype == "pedestrian" else (1.70 if atype == "autorickshaw" else (1.40 if atype == "motorcycle" else 1.50))),
                    "color": ACTOR_COLORS.get(atype, "#64748B"),
                    "is_ego": False
                }

    # 2. Extract timestamped frames
    frames = []
    events = []
    prev_action = "cruising"
    min_sep_so_far = float("inf")
    closest_approach_logged = False

    cv_forecaster = ConstantVelocityForecaster(pred_steps=30, dt=dt)
    calibrated_profile = getattr(policy, "profile", None)

    for idx, log in enumerate(step_logs):
        t = log["timestamp"]
        ego_x = log["ego_x"]
        ego_y = log["ego_y"]
        ego_spd = log["ego_speed"]
        accel = log["acceleration"]
        risk = log["risk_score"]
        action = log["action_type"]
        sep = log["min_separation"]

        # Track closest approach event
        if sep < min_sep_so_far:
            min_sep_so_far = sep
        elif not closest_approach_logged and min_sep_so_far < 4.0 and sep > min_sep_so_far:
            events.append({
                "timestamp": round(t, 2),
                "step": idx,
                "type": "CLOSEST_APPROACH",
                "label": f"Closest Approach: {min_sep_so_far:.2f}m",
                "severity": "high" if min_sep_so_far < 1.5 else "caution"
            })
            closest_approach_logged = True

        # Track action transition events
        if action != prev_action:
            if action == "emergency_braking":
                events.append({
                    "timestamp": round(t, 2),
                    "step": idx,
                    "type": "EMERGENCY_BRAKING",
                    "label": "Emergency Braking Activated",
                    "severity": "high"
                })
            elif action in ("advisory_braking", "proactive_braking", "uncertainty_caution"):
                events.append({
                    "timestamp": round(t, 2),
                    "step": idx,
                    "type": "PROACTIVE_BRAKE",
                    "label": f"Proactive Deceleration ({action})",
                    "severity": "caution"
                })
            prev_action = action

        # Build actor positions for this frame
        actors_in_frame = []
        for a in log["surrounding_actors"]:
            aid = a["actor_id"]
            # 3D coordinates: X=x, Y=0 (ground), Z=-y
            actors_in_frame.append({
                "id": aid,
                "x": float(a["x"]),
                "y": 0.0,
                "z": float(-a["y"]),
                "vx": float(a.get("vx", 0.0)),
                "vy": float(a.get("vy", 0.0)),
                "speed": float(a.get("speed", 0.0)),
                "heading": float(a.get("heading", 0.0))
            })

        # Generate future predictions issued at this step
        predictions_in_frame = {}
        for a in log["surrounding_actors"]:
            aid = a["actor_id"]
            # Constant velocity future rollout: 20 steps (2.0s)
            pred_steps_k = 20
            horizons = np.arange(1, pred_steps_k + 1) * dt
            pred_x = a["x"] + a.get("vx", 0.0) * horizons
            pred_y = a["y"] + a.get("vy", 0.0) * horizons

            # Convert to 3D trajectory points: [X, Y, Z]
            pts_3d = [[float(px), 0.1, float(-py)] for px, py in zip(pred_x, pred_y)]

            # Uncertainty ellipse estimation (grows with forecast horizon)
            if calibrated_profile is not None:
                sig_x = [float(value) for value in calibrated_profile["sigma_x"][:pred_steps_k]]
                sig_y = [float(value) for value in calibrated_profile["sigma_y"][:pred_steps_k]]
                uncertainty_label = "90% validation-calibrated marginal CV residual envelope"
            else:
                sig_x = [float(0.15 + 0.35 * h) for h in horizons]
                sig_y = [float(0.20 + 0.45 * h) for h in horizons]
                uncertainty_label = "illustrative forecast spread; not calibrated"

            predictions_in_frame[aid] = {
                "points": pts_3d,
                "horizons": [round(float(h), 2) for h in horizons],
                "sigma_x": sig_x,
                "sigma_y": sig_y,
                "rho": 0.0
                ,"uncertainty_label": uncertainty_label
            }

        frames.append({
            "step": idx,
            "timestamp": round(t, 2),
            "ego": {
                "x": float(ego_x),
                "y": 0.0,
                "z": float(-ego_y),
                "speed": float(ego_spd),
                "acceleration": float(accel),
                "heading": 0.0,
                "is_braking": bool(accel < -0.5)
            },
            "actors": actors_in_frame,
            "predictions": predictions_in_frame,
            "risk": {
                "score": float(risk),
                "min_separation": float(sep),
                "action_type": action
            }
        })

    # Add final outcome event
    if episode_result["collision"]:
        events.append({
            "timestamp": frames[-1]["timestamp"] if frames else 0.0,
            "step": len(frames) - 1,
            "type": "COLLISION",
            "label": "Collision Detected",
            "severity": "high"
        })
    elif episode_result["task_completed"]:
        events.append({
            "timestamp": frames[-1]["timestamp"] if frames else 0.0,
            "step": len(frames) - 1,
            "type": "COMPLETED",
            "label": "Goal Reached (Completed)",
            "severity": "normal"
        })

    return {
        "version": "1.0.0",
        "scenario_config": {
            "episode_id": episode_result["episode_id"],
            "scenario_type": episode_result["scenario_type"],
            "random_seed": episode_result["random_seed"],
            "policy_name": episode_result["policy_name"],
            "duration_seconds": episode_result["elapsed_time"],
            "time_step": dt,
            "total_frames": len(frames)
            ,"uncertainty_label": "90% validation-calibrated marginal CV residual envelope" if calibrated_profile is not None else "illustrative forecast spread; not calibrated"
        },
        "coordinate_mapping": {
            "description": "X_3d = X_sim, Y_3d = elevation, Z_3d = -Y_sim",
            "unit": "metres (SI)",
            "plane": "XZ ground plane"
        },
        "actors": list(actors_meta.values()),
        "frames": frames,
        "events": events,
        "summary": {
            "collision": episode_result["collision"],
            "near_collision": episode_result["near_collision"],
            "task_completed": episode_result["task_completed"],
            "min_separation": episode_result["min_separation"],
            "emergency_braking_count": episode_result["emergency_braking_count"],
            "elapsed_time": episode_result["elapsed_time"]
        }
    }
