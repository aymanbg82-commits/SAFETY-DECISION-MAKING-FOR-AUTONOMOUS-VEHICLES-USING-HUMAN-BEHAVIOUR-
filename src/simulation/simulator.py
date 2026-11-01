"""
Lightweight 2D Metric Traffic Simulator for Closed-Loop Safety Evaluation.
Simulates multi-agent traffic scenarios with SI metric units, explicit 10 Hz time steps,
exact OBB collision geometry, and pluggable safety policies.
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np

from src.simulation.geometry import OrientedBoundingBox, check_obb_collision, compute_obb_separation_distance
from src.simulation.policies import BasePolicy, ReactivePolicy
from src.data.synthetic_generator import SyntheticTrajectoryGenerator, ScenarioEpisode


class TrafficSimulator:
    """
    2D Closed-Loop Safety Simulator.
    Executes an episode while the ego vehicle is controlled by an active policy,
    evaluating collision, near-collision, task completion, and driving efficiency.
    """

    def __init__(
        self,
        dt: float = 0.1,
        goal_x: float = 50.0,
        max_duration_seconds: float = 6.5
    ):
        self.dt = dt
        self.goal_x = goal_x
        self.max_steps = int(max_duration_seconds / dt)
        self.near_collision_threshold = 1.5  # metres

    def run_episode(
        self,
        episode: ScenarioEpisode,
        policy: BasePolicy,
        forecaster_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run a single closed-loop simulation episode.

        Args:
            episode: ScenarioEpisode containing ground-truth non-ego trajectories.
            policy: BasePolicy instance (Reactive, PredictionAware, or UncertaintyAware).
            forecaster_model: Optional model used by prediction/uncertainty policies.

        Returns:
            Dict containing episode summary metrics and complete step-by-step logs.
        """
        # Separate ego initial state and non-ego trajectories
        ego_states_orig = [s for s in episode.states if s.is_ego]
        non_ego_states = [s for s in episode.states if not s.is_ego]

        if not ego_states_orig:
            raise ValueError(f"Episode {episode.episode_id} contains no ego vehicle.")

        # Organize non-ego states by time step
        other_actors_by_step: Dict[int, List[Dict[str, Any]]] = {}
        for s in non_ego_states:
            if s.step not in other_actors_by_step:
                other_actors_by_step[s.step] = []
            other_actors_by_step[s.step].append({
                "actor_id": s.actor_id,
                "actor_type": s.actor_type,
                "x": s.x,
                "y": s.y,
                "vx": s.vx,
                "vy": s.vy,
                "speed": s.speed,
                "heading": s.heading,
                "length": s.length,
                "width": s.width
            })

        # Initialize ego closed-loop state from initial state
        init_s = ego_states_orig[0]
        ego_x = init_s.x
        ego_y = init_s.y
        ego_vx = init_s.vx
        ego_vy = 0.0
        ego_speed = init_s.speed
        ego_heading = 0.0
        ego_len = init_s.length
        ego_width = init_s.width

        # Tracking metrics
        collision_occurred = False
        near_collision_occurred = False
        task_completed = False
        emergency_braking_count = 0
        min_separation_observed = float("inf")
        unnecessary_intervention = False

        step_logs: List[Dict[str, Any]] = []
        total_steps = min(self.max_steps, episode.total_steps)

        for step in range(total_steps):
            t = step * self.dt

            # Current ego state dictionary (fair observation, no future info)
            current_ego = {
                "step": step,
                "timestamp": t,
                "x": ego_x,
                "y": ego_y,
                "vx": ego_vx,
                "vy": ego_vy,
                "speed": ego_speed,
                "heading": ego_heading,
                "length": ego_len,
                "width": ego_width
            }

            surrounding = other_actors_by_step.get(step, [])

            # Compute policy action
            if hasattr(policy, "compute_action"):
                # Handle prediction / uncertainty policies
                import inspect
                sig = inspect.signature(policy.compute_action)
                if "forecaster_model" in sig.parameters:
                    action_result = policy.compute_action(current_ego, surrounding, forecaster_model=forecaster_model)
                else:
                    action_result = policy.compute_action(current_ego, surrounding)
            else:
                action_result = {"acceleration": 0.0, "risk_score": 0.0, "action_type": "cruising", "reason": "none"}

            accel = float(np.clip(action_result["acceleration"], -7.0, 3.5))
            risk_score = float(action_result.get("risk_score", 0.0))
            action_type = action_result.get("action_type", "cruising")

            if action_type == "emergency_braking":
                emergency_braking_count += 1

            # Exact OBB collision check at current step
            ego_box = OrientedBoundingBox(ego_x, ego_y, ego_len, ego_width, ego_heading)
            step_min_sep = float("inf")

            for actor in surrounding:
                act_box = OrientedBoundingBox(
                    actor["x"], actor["y"], actor["length"], actor["width"], actor["heading"]
                )
                if check_obb_collision(ego_box, act_box):
                    collision_occurred = True
                    step_min_sep = 0.0
                    break
                else:
                    sep = compute_obb_separation_distance(ego_box, act_box)
                    if sep < step_min_sep:
                        step_min_sep = sep

            if step_min_sep < self.near_collision_threshold and not collision_occurred:
                near_collision_occurred = True

            if step_min_sep < min_separation_observed:
                min_separation_observed = step_min_sep

            # Check if emergency action was taken when no danger was present (unnecessary intervention)
            if action_type == "emergency_braking" and step_min_sep > 6.0:
                unnecessary_intervention = True

            # Record step log
            step_logs.append({
                "step": step,
                "timestamp": round(t, 2),
                "ego_x": round(ego_x, 3),
                "ego_y": round(ego_y, 3),
                "ego_speed": round(ego_speed, 3),
                "acceleration": round(accel, 3),
                "risk_score": round(risk_score, 3),
                "action_type": action_type,
                "min_separation": round(step_min_sep, 3),
                "collision": collision_occurred,
                "surrounding_actors": surrounding
            })

            # Check termination conditions
            if collision_occurred:
                break

            if ego_x >= self.goal_x:
                task_completed = True
                break

            # Update ego closed-loop kinematics for next step
            ego_speed = max(0.0, ego_speed + accel * self.dt)
            ego_vx = ego_speed  # Assuming forward along lane
            ego_x += ego_vx * self.dt
            # ego_y remains controlled around centerline y=0

        elapsed_time = len(step_logs) * self.dt
        if min_separation_observed == float("inf"):
            min_separation_observed = 50.0

        return {
            "episode_id": episode.episode_id,
            "scenario_type": episode.scenario_type,
            "random_seed": episode.random_seed,
            "policy_name": policy.__class__.__name__,
            "collision": collision_occurred,
            "near_collision": near_collision_occurred,
            "task_completed": task_completed,
            "min_separation": round(min_separation_observed, 3),
            "emergency_braking_count": emergency_braking_count,
            "total_steps": len(step_logs),
            "elapsed_time": round(elapsed_time, 2),
            "unnecessary_intervention": unnecessary_intervention,
            "step_logs": step_logs
        }
