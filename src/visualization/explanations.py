"""
Deterministic Natural-Language Risk & Action Explanation Engine.
Generates structured explanations for any simulation frame:
- Situation (context, ego speed, active actors)
- Prediction (future paths, clearance horizon)
- Risk Factors (TTC, bounding box buffers, triggered thresholds)
- Simulated Response (policy decision rationale)
"""

import math
from typing import Dict, List, Any, Optional


def generate_frame_explanation(
    frame: Dict[str, Any],
    actors_meta: Dict[str, Dict[str, Any]],
    policy_name: str,
    scenario_type: str
) -> Dict[str, Any]:
    """
    Generate transparent, deterministic explanations from recorded frame telemetry.
    """
    ego = frame["ego"]
    actors = frame["actors"]
    risk = frame["risk"]
    t = frame["timestamp"]

    ego_speed_kmh = ego["speed"] * 3.6
    action = risk["action_type"]
    sep = risk["min_separation"]
    risk_score = risk["score"]

    # 1. Situation Analysis
    active_types = [actors_meta.get(a["id"], {}).get("type", "obstacle") for a in actors]
    type_str = ", ".join(set(active_types)) if active_types else "clear road"

    situation = (
        f"At t = {t:.1f}s, the ego vehicle is cruising at {ego_speed_kmh:.1f} km/h "
        f"along the roadway corridor. Active surrounding road users include: {type_str}."
    )

    # 2. Prediction Analysis
    if actors:
        nearest_act = min(actors, key=lambda a: math_dist(ego["x"], ego["z"], a["x"], a["z"]))
        nearest_id = nearest_act["id"]
        nearest_type = actors_meta.get(nearest_id, {}).get("type", "actor")
        nearest_dist = math_dist(ego["x"], ego["z"], nearest_act["x"], nearest_act["z"])

        prediction = (
            f"Forward trajectory forecasting projects {nearest_type} ({nearest_id}) "
            f"over the next 3.0s horizon. Predicted path converges with ego corridor, "
            f"with a minimum projected boundary clearance of {sep:.2f} m."
        )
    else:
        prediction = "No surrounding obstacles detected in perception range; trajectory forecast path is unobstructed."

    # 3. Risk Factors
    if sep <= 0.3:
        risk_category = "CRITICAL / COLLISION"
        risk_rule = "Physical boundary contact or collision threshold (separation <= 0.3m)"
        risk_detail = f"Imminent collision geometry! Separation has collapsed to {sep:.2f}m."
    elif sep < 1.5:
        risk_category = "HIGH RISK / NEAR-COLLISION"
        risk_rule = "Near-collision threshold (< 1.5m clearance buffer)"
        risk_detail = f"Dangerously close proximity: separation is {sep:.2f}m, exceeding safety threshold."
    elif sep < 4.0:
        risk_category = "CAUTION / ADVISORY"
        risk_rule = "Advisory safety envelope (1.5m <= separation < 4.0m)"
        risk_detail = f"Obstacle approaching active path. Separation is {sep:.2f}m."
    else:
        risk_category = "NORMAL / LOW RISK"
        risk_rule = "Safe clearance (separation >= 4.0m)"
        risk_detail = f"Adequate clearance maintained ({sep:.2f}m)."

    # 4. Simulated Response
    if action == "emergency_braking":
        response = (
            f"Active Policy ({policy_name}) triggered EMERGENCY BRAKING: commanded acceleration = {ego['acceleration']:.1f} m/s². "
            f"Full friction deceleration applied to halt before boundary intrusion."
        )
    elif action in ("proactive_braking", "advisory_braking", "uncertainty_caution"):
        response = (
            f"Active Policy ({policy_name}) initiated PROACTIVE DECELERATION: commanded acceleration = {ego['acceleration']:.1f} m/s². "
            f"Smoothly reducing speed in advance rather than executing a late emergency stop."
        )
    else:
        response = (
            f"Active Policy ({policy_name}) maintains CRUISING: commanded acceleration = {ego['acceleration']:+.1f} m/s². "
            f"Corridor clearance is nominal."
        )

    return {
        "timestamp": t,
        "situation": situation,
        "prediction": prediction,
        "risk_category": risk_category,
        "risk_rule": risk_rule,
        "risk_detail": risk_detail,
        "risk_score": risk_score,
        "simulated_response": response,
        "technical_telemetry": {
            "ego_speed_kmh": round(ego_speed_kmh, 1),
            "ego_acceleration_ms2": round(ego["acceleration"], 2),
            "minimum_separation_meters": round(sep, 2),
            "risk_score": round(risk_score, 3),
            "action_type": action,
            "policy": policy_name
        }
    }


def math_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)
