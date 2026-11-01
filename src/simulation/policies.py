"""
Safety Policies for Closed-Loop Traffic Simulation:
1. Reactive Baseline Policy (Distance & instantaneous TTC).
2. Point-Prediction-Aware Policy (Future trajectory point rollouts).
3. Uncertainty-Aware Policy (Covariance-augmented conservative clearance).
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np

from src.simulation.geometry import OrientedBoundingBox, check_obb_collision, compute_obb_separation_distance
from src.risk.geometric_risk import GeometricRiskEstimator


class BasePolicy:
    """Abstract interface for autonomous vehicle longitudinal control policy."""

    def __init__(
        self,
        target_speed: float = 11.11,    # ~40 km/h in m/s
        max_accel: float = 3.0,         # m/s^2
        emergency_brake: float = -6.5,  # m/s^2
        comfort_brake: float = -2.5     # m/s^2
    ):
        self.v_target = target_speed
        self.a_max = max_accel
        self.a_emerg = emergency_brake
        self.a_comf = comfort_brake
        self.risk_estimator = GeometricRiskEstimator()

    def compute_action(self, ego_state: Dict[str, Any], surrounding_actors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute control action.
        Returns dict containing:
            - 'acceleration': float (m/s^2)
            - 'risk_score': float [0, 1]
            - 'action_type': str ('cruising', 'advisory_braking', 'emergency_braking')
            - 'reason': str
        """
        raise NotImplementedError


class ReactivePolicy(BasePolicy):
    """
    Reactive Safety Policy Baseline.
    Uses only current observed distances and instantaneous TTC rules.
    Does NOT use future trajectory predictions.
    """

    def __init__(
        self,
        target_speed: float = 11.11,
        emergency_ttc: float = 1.8,
        warning_ttc: float = 3.2,
        emergency_dist: float = 2.0,
        warning_dist: float = 5.0
    ):
        super().__init__(target_speed=target_speed)
        self.ttc_emerg = emergency_ttc
        self.ttc_warn = warning_ttc
        self.d_emerg = emergency_dist
        self.d_warn = warning_dist

    def compute_action(self, ego_state: Dict[str, Any], surrounding_actors: List[Dict[str, Any]]) -> Dict[str, Any]:
        ego_box = OrientedBoundingBox(
            ego_state["x"], ego_state["y"], ego_state["length"], ego_state["width"], ego_state["heading"]
        )
        ego_vel = (ego_state["vx"], ego_state["vy"])

        max_risk = 0.0
        min_sep = float("inf")
        min_ttc: Optional[float] = None
        critical_actor_id = None

        for actor in surrounding_actors:
            act_box = OrientedBoundingBox(
                actor["x"], actor["y"], actor["length"], actor["width"], actor["heading"]
            )
            act_vel = (actor["vx"], actor["vy"])

            risk_eval = self.risk_estimator.evaluate_reactive_risk(
                ego_box, act_box, ego_vel, act_vel
            )

            r_score = risk_eval["risk_score"]
            sep = risk_eval["separation_distance"]
            ttc = risk_eval["ttc_seconds"]

            if sep < min_sep:
                min_sep = sep
            if ttc is not None and (min_ttc is None or ttc < min_ttc):
                min_ttc = ttc

            if r_score > max_risk:
                max_risk = r_score
                critical_actor_id = actor["actor_id"]

        # Action decision logic
        if min_sep <= self.d_emerg or (min_ttc is not None and min_ttc <= self.ttc_emerg):
            # Emergency braking
            accel = self.a_emerg
            action_type = "emergency_braking"
            reason = f"Critical reactive proximity (sep={min_sep:.1f}m, ttc={min_ttc})"
        elif min_sep <= self.d_warn or (min_ttc is not None and min_ttc <= self.ttc_warn):
            # Advisory braking
            accel = self.a_comf
            action_type = "advisory_braking"
            reason = f"Warning reactive proximity (sep={min_sep:.1f}m, ttc={min_ttc})"
        else:
            # Cruising towards target speed
            current_v = ego_state["speed"]
            accel = np.clip((self.v_target - current_v) * 0.8, -1.0, self.a_max)
            action_type = "cruising"
            reason = "Clear corridor"

        return {
            "acceleration": float(accel),
            "risk_score": float(max_risk),
            "action_type": action_type,
            "reason": reason,
            "min_separation": float(min_sep),
            "critical_actor_id": critical_actor_id
        }


class PredictionAwarePolicy(BasePolicy):
    """
    Point-Prediction-Aware Safety Policy.
    Uses future predicted point trajectories over horizon T_pred.
    Anticipates future conflicts and initiates proactive, comfortable deceleration.
    """

    def __init__(
        self,
        target_speed: float = 11.11,
        predicted_min_sep_crit: float = 1.8,
        predicted_min_sep_warn: float = 3.5,
        horizon_seconds: float = 3.0
    ):
        super().__init__(target_speed=target_speed)
        self.d_pred_crit = predicted_min_sep_crit
        self.d_pred_warn = predicted_min_sep_warn
        self.horizon = horizon_seconds

    def compute_action(
        self,
        ego_state: Dict[str, Any],
        surrounding_actors: List[Dict[str, Any]],
        forecaster_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Computes action using predicted future trajectories.
        If forecaster_model is provided, queries predictions;
        otherwise defaults to constant-velocity future projection.
        """
        ego_box = OrientedBoundingBox(
            ego_state["x"], ego_state["y"], ego_state["length"], ego_state["width"], ego_state["heading"]
        )
        ego_dim = (ego_state["length"], ego_state["width"])

        # Ego nominal constant velocity rollout over 30 steps (3.0s)
        dt = 0.1
        T_pred = 30
        horizons = np.arange(1, T_pred + 1) * dt
        ego_future_traj = np.column_stack([
            ego_state["x"] + ego_state["vx"] * horizons,
            ego_state["y"] + ego_state["vy"] * horizons
        ])

        max_risk = 0.0
        min_pred_dist = float("inf")
        closest_pred_time = None
        critical_actor_id = None
        predicted_collision = False

        for actor in surrounding_actors:
            act_dim = (actor["length"], actor["width"])

            # Obtain predicted future trajectory
            if "predicted_future" in actor and actor["predicted_future"] is not None:
                act_future = actor["predicted_future"]
            else:
                # Constant velocity fallback projection
                act_future = np.column_stack([
                    actor["x"] + actor["vx"] * horizons,
                    actor["y"] + actor["vy"] * horizons
                ])

            pred_eval = self.risk_estimator.evaluate_predicted_risk(
                ego_future_traj, act_future, ego_dim, act_dim
            )

            p_risk = pred_eval["predicted_risk_score"]
            p_dist = pred_eval["min_predicted_distance"]

            if p_dist < min_pred_dist:
                min_pred_dist = p_dist
                closest_pred_time = pred_eval["predicted_time_to_closest"]
                critical_actor_id = actor["actor_id"]

            if pred_eval["predicted_collision"]:
                predicted_collision = True

            if p_risk > max_risk:
                max_risk = p_risk

        # Current immediate separation check (safety fallback)
        immediate_min_sep = float("inf")
        for actor in surrounding_actors:
            act_box = OrientedBoundingBox(actor["x"], actor["y"], actor["length"], actor["width"], actor["heading"])
            sep = compute_obb_separation_distance(ego_box, act_box)
            if sep < immediate_min_sep:
                immediate_min_sep = sep

        # Prediction-aware decision logic
        if immediate_min_sep < 1.5 or (predicted_collision and closest_pred_time is not None and closest_pred_time < 1.2):
            accel = self.a_emerg
            action_type = "emergency_braking"
            reason = f"Imminent predicted collision (min_dist={min_pred_dist:.1f}m in {closest_pred_time}s)"
        elif min_pred_dist < self.d_pred_crit or predicted_collision:
            # Proactive comfortable deceleration
            accel = self.a_comf * 1.2
            action_type = "proactive_braking"
            reason = f"Predicted critical clearance violation (dist={min_pred_dist:.1f}m)"
        elif min_pred_dist < self.d_pred_warn:
            # Gentle advisory speed adjustment
            accel = self.a_comf * 0.6
            action_type = "advisory_braking"
            reason = f"Predicted advisory proximity (dist={min_pred_dist:.1f}m)"
        else:
            current_v = ego_state["speed"]
            accel = np.clip((self.v_target - current_v) * 0.8, -1.0, self.a_max)
            action_type = "cruising"
            reason = "Clear predicted path"

        return {
            "acceleration": float(accel),
            "risk_score": float(max_risk),
            "action_type": action_type,
            "reason": reason,
            "min_predicted_distance": float(min_pred_dist),
            "critical_actor_id": critical_actor_id
        }


class UncertaintyAwarePolicy(BasePolicy):
    """
    Uncertainty-Aware Safety Policy.
    Uses predicted trajectory distributions (mean and covariance ellipses).
    When future prediction variance is high (e.g. cut-in maneuver or hesitating pedestrian),
    expands the safety margin dynamically, braking earlier and smoother.
    When prediction variance is low, maintains high efficiency without phantom braking.
    """

    def __init__(
        self,
        target_speed: float = 11.11,
        conservative_quantile: float = 0.90,
        safety_clearance_buffer: float = 2.0
    ):
        super().__init__(target_speed=target_speed)
        self.quantile = conservative_quantile
        self.buffer = safety_clearance_buffer

    def compute_action(
        self,
        ego_state: Dict[str, Any],
        surrounding_actors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        ego_box = OrientedBoundingBox(
            ego_state["x"], ego_state["y"], ego_state["length"], ego_state["width"], ego_state["heading"]
        )
        ego_dim = (ego_state["length"], ego_state["width"])

        dt = 0.1
        T_pred = 30
        horizons = np.arange(1, T_pred + 1) * dt
        ego_future_traj = np.column_stack([
            ego_state["x"] + ego_state["vx"] * horizons,
            ego_state["y"] + ego_state["vy"] * horizons
        ])

        max_risk = 0.0
        min_conservative_dist = float("inf")
        critical_actor_id = None

        for actor in surrounding_actors:
            act_dim = (actor["length"], actor["width"])

            # Check if actor has predicted distribution
            if "predicted_mu" in actor and "predicted_log_sigma" in actor:
                mu = actor["predicted_mu"]
                log_sigma = actor["predicted_log_sigma"]
                rho = actor.get("predicted_rho", np.zeros(len(mu)))
            else:
                # Fallback: Constant velocity mean with growing heuristic variance
                mu = np.column_stack([
                    actor["x"] + actor["vx"] * horizons,
                    actor["y"] + actor["vy"] * horizons
                ])
                # Heuristic growing uncertainty: sigma(t) = 0.2 + 0.3 * t
                sig_growth = np.log(0.2 + 0.35 * horizons)
                log_sigma = np.column_stack([sig_growth, sig_growth])
                rho = np.zeros(len(horizons))

            unc_eval = self.risk_estimator.evaluate_uncertainty_risk(
                ego_future_traj, mu, log_sigma, rho, ego_dim, act_dim, quantile=self.quantile
            )

            u_risk = unc_eval["uncertainty_risk_score"]
            c_dist = unc_eval["conservative_min_distance"]

            if c_dist < min_conservative_dist:
                min_conservative_dist = c_dist
                critical_actor_id = actor["actor_id"]

            if u_risk > max_risk:
                max_risk = u_risk

        # Immediate current distance fallback
        immediate_min_sep = float("inf")
        for actor in surrounding_actors:
            act_box = OrientedBoundingBox(actor["x"], actor["y"], actor["length"], actor["width"], actor["heading"])
            sep = compute_obb_separation_distance(ego_box, act_box)
            if sep < immediate_min_sep:
                immediate_min_sep = sep

        # Uncertainty-aware control decision
        if immediate_min_sep < 1.5:
            accel = self.a_emerg
            action_type = "emergency_braking"
            reason = f"Immediate physical encroachment (sep={immediate_min_sep:.1f}m)"
        elif min_conservative_dist <= 0.2:
            # Conservative boundary overlaps ego path: apply firm smooth deceleration
            accel = self.a_comf * 1.3
            action_type = "proactive_braking"
            reason = f"Conservative 90% hazard envelope overlap (d_cons={min_conservative_dist:.1f}m)"
        elif min_conservative_dist < self.buffer:
            # Gentle pacing / cautionary easing
            accel = self.a_comf * 0.7
            action_type = "uncertainty_caution"
            reason = f"Uncertainty buffer proximity (d_cons={min_conservative_dist:.1f}m)"
        else:
            current_v = ego_state["speed"]
            accel = np.clip((self.v_target - current_v) * 0.8, -1.0, self.a_max)
            action_type = "cruising"
            reason = "Confidence corridor clear"

        return {
            "acceleration": float(accel),
            "risk_score": float(max_risk),
            "action_type": action_type,
            "reason": reason,
            "conservative_distance": float(min_conservative_dist),
            "critical_actor_id": critical_actor_id
        }


class CalibratedUncertaintyPolicy(BasePolicy):
    """Uncertainty policy using a validation-calibrated CV residual envelope."""

    def __init__(self, calibration_profile: Dict[str, Any], target_speed: float = 11.11):
        super().__init__(target_speed=target_speed)
        self.profile = calibration_profile
        self.quantile = float(calibration_profile["quantile"])
        self.horizons = np.asarray(calibration_profile["horizons"], dtype=float)
        self.sigma_x = np.asarray(calibration_profile["sigma_x"], dtype=float)
        self.sigma_y = np.asarray(calibration_profile["sigma_y"], dtype=float)
        self.rho = np.asarray(calibration_profile["rho"], dtype=float)
        self.buffer = 2.0

    def compute_action(self, ego_state: Dict[str, Any], surrounding_actors: List[Dict[str, Any]]) -> Dict[str, Any]:
        ego_box = OrientedBoundingBox(ego_state["x"], ego_state["y"], ego_state["length"], ego_state["width"], ego_state["heading"])
        ego_future = np.column_stack([
            ego_state["x"] + ego_state["vx"] * self.horizons,
            ego_state["y"] + ego_state["vy"] * self.horizons,
        ])
        min_conservative_dist = float("inf")
        max_risk = 0.0
        critical_actor_id = None
        for actor in surrounding_actors:
            actor_future = np.column_stack([
                actor["x"] + actor["vx"] * self.horizons,
                actor["y"] + actor["vy"] * self.horizons,
            ])
            evaluation = self.risk_estimator.evaluate_uncertainty_risk(
                ego_future,
                actor_future,
                np.column_stack([np.log(self.sigma_x), np.log(self.sigma_y)]),
                self.rho,
                (ego_state["length"], ego_state["width"]),
                (actor["length"], actor["width"]),
                quantile=self.quantile,
            )
            if evaluation["conservative_min_distance"] < min_conservative_dist:
                min_conservative_dist = evaluation["conservative_min_distance"]
                critical_actor_id = actor["actor_id"]
            max_risk = max(max_risk, evaluation["uncertainty_risk_score"])

        immediate_min_sep = min(
            (compute_obb_separation_distance(ego_box, OrientedBoundingBox(a["x"], a["y"], a["length"], a["width"], a["heading"])) for a in surrounding_actors),
            default=float("inf"),
        )
        if immediate_min_sep < 1.5:
            acceleration, action_type = self.a_emerg, "emergency_braking"
            reason = f"Immediate physical encroachment (sep={immediate_min_sep:.1f}m)"
        elif min_conservative_dist <= 0.2:
            acceleration, action_type = self.a_comf * 1.2, "proactive_braking"
            reason = f"Validation-calibrated envelope overlap (d_cons={min_conservative_dist:.1f}m)"
        elif min_conservative_dist < self.buffer:
            acceleration, action_type = self.a_comf * 0.7, "uncertainty_caution"
            reason = f"Validation-calibrated uncertainty buffer (d_cons={min_conservative_dist:.1f}m)"
        else:
            acceleration = np.clip((self.v_target - ego_state["speed"]) * 0.8, -1.0, self.a_max)
            action_type, reason = "cruising", "Validation-calibrated corridor clear"
        return {
            "acceleration": float(acceleration),
            "risk_score": float(max_risk),
            "action_type": action_type,
            "reason": reason,
            "conservative_distance": float(min_conservative_dist),
            "critical_actor_id": critical_actor_id,
        }
