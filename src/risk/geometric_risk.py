"""
Geometric Risk Estimation, Time-To-Collision (TTC),
Predicted Minimum Separation, and Uncertainty-Aware Hazard Scoring.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

from src.simulation.geometry import (
    OrientedBoundingBox,
    check_obb_collision,
    compute_obb_separation_distance,
    calculate_physical_ttc
)


class GeometricRiskEstimator:
    """
    Computes deterministic geometric risk, predicted minimum separation,
    and uncertainty-augmented collision probabilities.
    """

    def __init__(
        self,
        collision_dist_threshold: float = 0.3,
        near_collision_threshold: float = 1.5,
        ttc_critical: float = 2.0,
        ttc_warning: float = 3.5,
        risk_horizon: float = 3.0
    ):
        self.d_collision = collision_dist_threshold
        self.d_near = near_collision_threshold
        self.ttc_crit = ttc_critical
        self.ttc_warn = ttc_warning
        self.horizon = risk_horizon

    def evaluate_reactive_risk(
        self,
        ego_box: OrientedBoundingBox,
        actor_box: OrientedBoundingBox,
        ego_vel: Tuple[float, float],
        actor_vel: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Evaluate reactive risk based strictly on current observed positions and instantaneous TTC.
        """
        is_collision = check_obb_collision(ego_box, actor_box)
        current_sep = 0.0 if is_collision else compute_obb_separation_distance(ego_box, actor_box)
        is_near = (current_sep < self.d_near)

        ttc = calculate_physical_ttc(
            ego_pos=(ego_box.x, ego_box.y),
            ego_vel=ego_vel,
            actor_pos=(actor_box.x, actor_box.y),
            actor_vel=actor_vel,
            ego_dim=(ego_box.length, ego_box.width),
            actor_dim=(actor_box.length, actor_box.width)
        )

        # Reactive risk score in [0, 1]
        if is_collision:
            risk_score = 1.0
        elif ttc is not None and ttc < self.ttc_crit:
            # High risk ramping from 0.7 to 1.0
            risk_score = 0.7 + 0.3 * (1.0 - max(0.0, ttc) / self.ttc_crit)
        elif ttc is not None and ttc < self.ttc_warn:
            # Medium warning risk ramping from 0.3 to 0.7
            risk_score = 0.3 + 0.4 * (1.0 - (ttc - self.ttc_crit) / (self.ttc_warn - self.ttc_crit))
        elif current_sep < self.d_near:
            # Proximity risk
            risk_score = 0.5 * (1.0 - current_sep / self.d_near)
        else:
            risk_score = 0.0

        return {
            "risk_score": float(np.clip(risk_score, 0.0, 1.0)),
            "separation_distance": float(current_sep),
            "ttc_seconds": float(ttc) if ttc is not None else None,
            "is_collision": is_collision,
            "is_near_collision": is_near
        }

    def evaluate_predicted_risk(
        self,
        ego_future_traj: np.ndarray,      # (T_pred, 2) absolute (x, y) coordinates
        actor_future_traj: np.ndarray,    # (T_pred, 2) absolute (x, y) coordinates
        ego_dim: Tuple[float, float],     # (length, width)
        actor_dim: Tuple[float, float]    # (length, width)
    ) -> Dict[str, Any]:
        """
        Evaluate collision risk using forward-predicted point trajectories over horizon T_pred.
        Computes minimum predicted separation distance and predicted time-to-closest-approach.
        """
        T_pred = min(len(ego_future_traj), len(actor_future_traj))
        min_distance = float("inf")
        closest_step = -1
        collision_step = -1

        dt = 0.1
        ego_len, ego_w = ego_dim
        act_len, act_w = actor_dim

        for step in range(T_pred):
            ex, ey = ego_future_traj[step]
            ax, ay = actor_future_traj[step]

            # Approximate heading from consecutive displacements if available
            e_heading = 0.0
            if step > 0:
                e_heading = math.atan2(ey - ego_future_traj[step - 1, 1], ex - ego_future_traj[step - 1, 0])

            a_heading = 0.0
            if step > 0:
                a_heading = math.atan2(ay - actor_future_traj[step - 1, 1], ax - actor_future_traj[step - 1, 0])

            box_e = OrientedBoundingBox(ex, ey, ego_len, ego_w, e_heading)
            box_a = OrientedBoundingBox(ax, ay, act_len, act_w, a_heading)

            if check_obb_collision(box_e, box_a):
                if collision_step == -1:
                    collision_step = step
                dist = 0.0
            else:
                dist = compute_obb_separation_distance(box_e, box_a)

            if dist < min_distance:
                min_distance = dist
                closest_step = step

        time_to_closest = (closest_step + 1) * dt if closest_step != -1 else None
        predicted_collision = (collision_step != -1)

        # Risk score mapping
        if predicted_collision:
            # Escalates higher the sooner collision occurs
            time_to_coll = (collision_step + 1) * dt
            pred_risk = 0.8 + 0.2 * (1.0 - min(time_to_coll, self.horizon) / self.horizon)
        elif min_distance < self.d_near:
            pred_risk = 0.4 + 0.4 * (1.0 - min_distance / self.d_near)
        elif min_distance < 4.0:
            pred_risk = 0.1 + 0.3 * (1.0 - (min_distance - self.d_near) / (4.0 - self.d_near))
        else:
            pred_risk = 0.0

        return {
            "predicted_risk_score": float(np.clip(pred_risk, 0.0, 1.0)),
            "min_predicted_distance": float(min_distance),
            "predicted_time_to_closest": float(time_to_closest) if time_to_closest is not None else None,
            "predicted_collision": predicted_collision
        }

    def evaluate_uncertainty_risk(
        self,
        ego_future_traj: np.ndarray,      # (T_pred, 2)
        actor_mu: np.ndarray,             # (T_pred, 2) mean (x, y)
        actor_log_sigma: np.ndarray,      # (T_pred, 2) log(sigma_x, sigma_y)
        actor_rho: np.ndarray,            # (T_pred,) correlation rho
        ego_dim: Tuple[float, float],
        actor_dim: Tuple[float, float],
        quantile: float = 0.90
    ) -> Dict[str, Any]:
        """
        Evaluate uncertainty-aware risk by projecting the conservative (1 - alpha)
        confidence boundary of the predicted trajectory towards ego vehicle.
        """
        T_pred = min(len(ego_future_traj), len(actor_mu))
        chi2_crit = -2.0 * math.log(1.0 - quantile)
        scale_factor = math.sqrt(chi2_crit)

        min_conservative_dist = float("inf")
        closest_step = -1

        for step in range(T_pred):
            ex, ey = ego_future_traj[step]
            mx, my = actor_mu[step]
            sx = math.exp(actor_log_sigma[step, 0])
            sy = math.exp(actor_log_sigma[step, 1])

            # Unit vector from actor mean to ego
            dir_x = ex - mx
            dir_y = ey - my
            center_dist = math.hypot(dir_x, dir_y)

            if center_dist > 1e-4:
                ux = dir_x / center_dist
                uy = dir_y / center_dist
            else:
                ux, uy = 1.0, 0.0

            # Directional standard deviation along line towards ego:
            # sigma_dir = sqrt(u^T Sigma u)
            rho_val = float(np.clip(actor_rho[step] if actor_rho.ndim > 0 else actor_rho, -0.99, 0.99))
            cov_xy = rho_val * sx * sy
            var_dir = (ux * sx)**2 + 2.0 * ux * uy * cov_xy + (uy * sy)**2
            sigma_dir = math.sqrt(max(1e-6, var_dir))

            # Conservative boundary offset towards ego
            effective_clearance = max(0.0, center_dist - scale_factor * sigma_dir)
            # Subtract physical bounding radii
            r_ego = math.hypot(ego_dim[0] / 2.0, ego_dim[1] / 2.0)
            r_act = math.hypot(actor_dim[0] / 2.0, actor_dim[1] / 2.0)
            conservative_box_dist = max(0.0, effective_clearance - (r_ego + r_act))

            if conservative_box_dist < min_conservative_dist:
                min_conservative_dist = conservative_box_dist
                closest_step = step

        # Uncertainty risk score
        if min_conservative_dist <= 0.0:
            unc_risk = 0.95
        elif min_conservative_dist < self.d_near:
            unc_risk = 0.6 + 0.35 * (1.0 - min_conservative_dist / self.d_near)
        elif min_conservative_dist < 4.0:
            unc_risk = 0.2 + 0.4 * (1.0 - (min_conservative_dist - self.d_near) / (4.0 - self.d_near))
        else:
            unc_risk = 0.0

        return {
            "uncertainty_risk_score": float(np.clip(unc_risk, 0.0, 1.0)),
            "conservative_min_distance": float(min_conservative_dist),
            "critical_step": int(closest_step),
            "confidence_quantile": float(quantile)
        }
