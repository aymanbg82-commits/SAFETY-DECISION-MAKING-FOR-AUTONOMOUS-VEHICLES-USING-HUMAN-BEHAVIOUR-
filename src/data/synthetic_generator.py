"""
Synthetic Trajectory and Traffic Episode Generator.
Generates reproducible, parameterized multi-agent traffic scenarios
with metric coordinates (SI metres, 10 Hz dt = 0.1s), explicit actor dimensions,
and Indian-road-inspired behavioral archetypes.
"""

import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from src.simulation.geometry import OrientedBoundingBox, check_obb_collision, compute_obb_separation_distance


@dataclass
class ActorState:
    step: int
    timestamp: float          # seconds
    actor_id: str
    actor_type: str
    x: float                  # metres
    y: float                  # metres
    vx: float                 # m/s
    vy: float                 # m/s
    speed: float              # m/s
    heading: float            # radians
    length: float             # metres
    width: float              # metres
    is_ego: bool


@dataclass
class ScenarioEpisode:
    episode_id: str
    scenario_type: str
    random_seed: int
    time_step: float
    total_steps: int
    parameters: Dict[str, Any]
    states: List[ActorState]
    is_synthetic: bool = True
    data_label: str = "SYNTHETIC_INDIAN_ROAD_SIMULATION"


# Standard dimensions and dynamics for Indian road user types (SI units)
ACTOR_SPECS = {
    "car": {"length": 4.5, "width": 2.0, "max_accel": 3.5, "max_decel": 7.0},
    "motorcycle": {"length": 2.2, "width": 0.9, "max_accel": 4.5, "max_decel": 6.5},
    "autorickshaw": {"length": 2.8, "width": 1.3, "max_accel": 2.5, "max_decel": 5.5},
    "truck": {"length": 7.5, "width": 2.5, "max_accel": 2.0, "max_decel": 4.5},
    "pedestrian": {"length": 0.6, "width": 0.6, "max_accel": 2.0, "max_decel": 3.0},
}


class SyntheticTrajectoryGenerator:
    """
    Generator for parameterized traffic scenarios inspired by Indian road driving conditions:
    1. Pedestrian Crossing (with mid-street hesitation / variable speed).
    2. Auto-rickshaw Cut-In (three-wheeler lateral merge).
    3. Sudden Animal Entry (synthetic lateral entry from a roadside boundary).
    4. Mixed-Traffic Intersection (unregulated junction with autorickshaw and cross-traffic).
    5. Unregulated Overtaking (slow lead vehicle with oncoming traffic).
    """

    def __init__(self, time_step: float = 0.1, default_seed: int = 42):
        self.dt = time_step
        self.default_seed = default_seed

    def generate_episode(
        self,
        scenario_type: str,
        seed: Optional[int] = None,
        duration_seconds: float = 6.0,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> ScenarioEpisode:
        """
        Generate a complete multi-agent traffic episode.
        """
        if seed is None:
            seed = self.default_seed
        rng = np.random.RandomState(seed)

        total_steps = int(duration_seconds / self.dt)
        episode_id = f"ep_{scenario_type}_{seed:05d}"
        params = custom_params or {}

        if scenario_type == "pedestrian_crossing":
            states, resolved_params = self._generate_pedestrian_crossing(rng, total_steps, params)
        elif scenario_type in ("motorcycle_cut_in", "autorickshaw_cut_in"):
            if scenario_type == "autorickshaw_cut_in":
                states, resolved_params = self._generate_autorickshaw_cut_in(rng, total_steps, params)
            else:
                states, resolved_params = self._generate_motorcycle_cut_in(rng, total_steps, params)
        elif scenario_type == "animal_entry":
            states, resolved_params = self._generate_animal_entry(rng, total_steps, params)
        elif scenario_type == "mixed_traffic_intersection":
            states, resolved_params = self._generate_mixed_traffic_intersection(rng, total_steps, params)
        elif scenario_type == "unregulated_overtaking":
            states, resolved_params = self._generate_unregulated_overtaking(rng, total_steps, params)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        return ScenarioEpisode(
            episode_id=episode_id,
            scenario_type=scenario_type,
            random_seed=seed,
            time_step=self.dt,
            total_steps=total_steps,
            parameters=resolved_params,
            states=states,
            is_synthetic=True,
            data_label="SYNTHETIC_INDIAN_ROAD_SIMULATION"
        )

    def _generate_pedestrian_crossing(
        self, rng: np.random.RandomState, total_steps: int, params: Dict[str, Any]
    ) -> Tuple[List[ActorState], Dict[str, Any]]:
        """
        Scenario 1: Pedestrian Crossing.
        Ego cruises along main road (lane y=0). A pedestrian steps out from curb
        and crosses perpendicular to road, with potential hesitation.
        """
        ego_speed = params.get("ego_speed", rng.uniform(8.5, 11.5))       # m/s (~30-40 km/h)
        ped_speed = params.get("ped_speed", rng.uniform(1.0, 1.6))         # m/s
        ped_start_distance = params.get("ped_start_distance")
        ped_start_x = params.get("ped_start_x", rng.uniform(20.0, 32.0))   # road-frame x, retained for compatibility
        crossing_time = params.get("crossing_time", 0.0)                   # seconds after episode start
        ped_from_left = params.get("ped_from_left", rng.choice([True, False]))
        has_hesitation = params.get("has_hesitation", rng.uniform(0, 1) < 0.40)
        hesitation_step = int(rng.uniform(15, 28)) if has_hesitation else -1

        resolved_params = {
            "ego_speed": float(ego_speed),
            "ped_speed": float(ped_speed),
            "ped_start_distance": float(ped_start_distance if ped_start_distance is not None else ped_start_x + 15.0),
            "ped_start_x": float(ped_start_x),
            "crossing_time": float(crossing_time),
            "ped_from_left": bool(ped_from_left),
            "has_hesitation": bool(has_hesitation),
            "hesitation_step": int(hesitation_step)
        }

        states = []
        ego_spec = ACTOR_SPECS["car"]
        ped_spec = ACTOR_SPECS["pedestrian"]

        ego_x = -15.0
        ego_y = 0.0
        # The public control is a gap from the ego's initial position. Keep the
        # road-frame coordinate in the episode metadata for existing consumers.
        if ped_start_distance is not None:
            ped_x = ego_x + float(ped_start_distance)
            resolved_params["ped_start_x"] = float(ped_x)
        else:
            ped_x = ped_start_x
        ped_y = -3.8 if ped_from_left else 3.8
        ped_dir_y = 1.0 if ped_from_left else -1.0

        for s in range(total_steps):
            t = s * self.dt

            # Ego kinematics (cruising with slight realistic noise)
            ego_vx = ego_speed + rng.normal(0, 0.05)
            ego_vy = 0.0
            ego_heading = 0.0
            ego_x += ego_vx * self.dt

            states.append(ActorState(
                step=s, timestamp=t, actor_id="ego", actor_type="car",
                x=ego_x, y=ego_y, vx=ego_vx, vy=ego_vy,
                speed=ego_vx, heading=ego_heading,
                length=ego_spec["length"], width=ego_spec["width"], is_ego=True
            ))

            # Pedestrian kinematics
            crossing_started = t >= float(crossing_time)
            is_hesitating = (crossing_started and has_hesitation and hesitation_step <= s < hesitation_step + 12)
            current_ped_speed = (0.2 if is_hesitating else ped_speed) if crossing_started else 0.0
            ped_vy = current_ped_speed * ped_dir_y
            ped_vx = rng.normal(0, 0.02)
            ped_y += ped_vy * self.dt
            ped_x += ped_vx * self.dt
            ped_heading = math.pi / 2 if ped_dir_y > 0 else -math.pi / 2

            states.append(ActorState(
                step=s, timestamp=t, actor_id="ped_01", actor_type="pedestrian",
                x=ped_x, y=ped_y, vx=ped_vx, vy=ped_vy,
                speed=math.hypot(ped_vx, ped_vy), heading=ped_heading,
                length=ped_spec["length"], width=ped_spec["width"], is_ego=False
            ))

        return states, resolved_params

    def _generate_animal_entry(
        self, rng: np.random.RandomState, total_steps: int, params: Dict[str, Any]
    ) -> Tuple[List[ActorState], Dict[str, Any]]:
        """Generate a simple synthetic animal entering laterally after a delay."""
        ego_speed = float(params.get("ego_speed", rng.uniform(8.0, 11.0)))
        animal_start_distance = float(params.get("animal_start_distance", 28.0))
        animal_entry_time = float(params.get("animal_entry_time", 1.5))
        animal_direction = float(params.get("animal_direction", -1.0))
        animal_speed = float(params.get("animal_speed", 2.0))
        animal_length = float(params.get("animal_length", 1.2))
        animal_width = float(params.get("animal_width", 0.7))
        animal_start_y = float(params.get("animal_start_y", 4.5 if animal_direction < 0 else -4.5))
        entry_step = int(round(animal_entry_time / self.dt))

        resolved_params = {
            "ego_speed": ego_speed,
            "animal_start_distance": animal_start_distance,
            "animal_entry_time": animal_entry_time,
            "animal_direction": animal_direction,
            "animal_speed": animal_speed,
            "animal_length": animal_length,
            "animal_width": animal_width,
            "animal_start_y": animal_start_y,
        }

        states = []
        ego_spec = ACTOR_SPECS["car"]
        ego_x = -15.0
        animal_x = ego_x + animal_start_distance
        animal_y = animal_start_y
        direction = 1.0 if animal_direction >= 0 else -1.0

        for s in range(total_steps):
            t = s * self.dt
            ego_speed_step = ego_speed + rng.normal(0.0, 0.05)
            ego_x += ego_speed_step * self.dt
            states.append(ActorState(
                step=s, timestamp=t, actor_id="ego", actor_type="car",
                x=ego_x, y=0.0, vx=ego_speed_step, vy=0.0, speed=ego_speed_step,
                heading=0.0, length=ego_spec["length"], width=ego_spec["width"], is_ego=True
            ))

            entered = s >= entry_step
            animal_vy = direction * animal_speed if entered and abs(animal_y) > 0.01 else 0.0
            animal_x += rng.normal(0.0, 0.01) * self.dt
            animal_y += animal_vy * self.dt
            if (direction > 0 and animal_y > 0.0) or (direction < 0 and animal_y < 0.0):
                animal_y = 0.0
                animal_vy = 0.0
            states.append(ActorState(
                step=s, timestamp=t, actor_id="animal_01", actor_type="animal",
                x=animal_x, y=animal_y, vx=0.0, vy=animal_vy,
                speed=abs(animal_vy),
                heading=math.pi / 2 if direction > 0 else -math.pi / 2,
                length=animal_length, width=animal_width, is_ego=False
            ))

        return states, resolved_params

    def _generate_autorickshaw_cut_in(
        self, rng: np.random.RandomState, total_steps: int, params: Dict[str, Any]
    ) -> Tuple[List[ActorState], Dict[str, Any]]:
        """Generate an autorickshaw merging laterally into the ego lane."""
        ego_speed = params.get("ego_speed", rng.uniform(8.0, 11.0))
        auto_speed = params.get("cut_in_speed", ego_speed + rng.uniform(-1.0, 1.5))
        cut_in_step = int(params.get("cut_in_step", int(rng.uniform(12, 22))))
        lateral_offset = float(params.get("lateral_offset", 2.5))
        lateral_speed = float(params.get("lateral_speed", 1.2))
        initial_gap = float(params.get("cut_in_gap", 10.0))

        resolved_params = {
            "ego_speed": float(ego_speed),
            "cut_in_speed": float(auto_speed),
            "cut_in_step": cut_in_step,
            "lateral_offset": lateral_offset,
            "lateral_speed": lateral_speed,
            "cut_in_gap": initial_gap,
        }

        states = []
        ego_spec = ACTOR_SPECS["car"]
        auto_spec = ACTOR_SPECS["autorickshaw"]
        ego_x = 0.0
        auto_x = initial_gap
        auto_y = lateral_offset
        direction = -1.0 if lateral_offset > 0 else 1.0

        for s in range(total_steps):
            t = s * self.dt
            ego_x += ego_speed * self.dt
            states.append(ActorState(
                step=s, timestamp=t, actor_id="ego", actor_type="car",
                x=ego_x, y=0.0, vx=ego_speed, vy=0.0, speed=ego_speed,
                heading=0.0, length=ego_spec["length"], width=ego_spec["width"], is_ego=True
            ))

            if s < cut_in_step:
                auto_vx = auto_speed
                auto_vy = 0.0
            elif abs(auto_y) > 0.01:
                auto_vx = auto_speed
                auto_vy = direction * lateral_speed
            else:
                auto_vx = auto_speed
                auto_vy = 0.0
                auto_y = 0.0

            auto_x += auto_vx * self.dt
            auto_y += auto_vy * self.dt
            if (direction < 0 and auto_y < 0) or (direction > 0 and auto_y > 0):
                auto_y = 0.0
                auto_vy = 0.0
            states.append(ActorState(
                step=s, timestamp=t, actor_id="auto_01", actor_type="autorickshaw",
                x=auto_x, y=auto_y, vx=auto_vx, vy=auto_vy,
                speed=math.hypot(auto_vx, auto_vy),
                heading=math.atan2(auto_vy, auto_vx) if auto_vy else 0.0,
                length=auto_spec["length"], width=auto_spec["width"], is_ego=False
            ))

        return states, resolved_params

    def _generate_motorcycle_cut_in(
        self, rng: np.random.RandomState, total_steps: int, params: Dict[str, Any]
    ) -> Tuple[List[ActorState], Dict[str, Any]]:
        """
        Scenario 2: Motorcycle Cut-In (Lane Filtering).
        A two-wheeler filtering between traffic cuts sharply in front of the ego vehicle.
        """
        ego_speed = params.get("ego_speed", rng.uniform(8.0, 11.0))
        mc_speed = params.get("mc_speed", ego_speed + rng.uniform(1.5, 3.5))
        cut_in_trigger_step = params.get("cut_in_step", int(rng.uniform(12, 22)))
        cut_in_from_right = params.get("from_right", rng.choice([True, False]))
        initial_lateral_offset = 2.0 if cut_in_from_right else -2.0
        longitudinal_gap_at_cut = params.get("cut_gap", rng.uniform(5.0, 9.0))

        resolved_params = {
            "ego_speed": float(ego_speed),
            "mc_speed": float(mc_speed),
            "cut_in_step": int(cut_in_trigger_step),
            "cut_in_from_right": bool(cut_in_from_right),
            "longitudinal_gap": float(longitudinal_gap_at_cut)
        }

        states = []
        ego_spec = ACTOR_SPECS["car"]
        mc_spec = ACTOR_SPECS["motorcycle"]

        ego_x = 0.0
        ego_y = 0.0
        mc_x = longitudinal_gap_at_cut - 2.0
        mc_y = initial_lateral_offset

        for s in range(total_steps):
            t = s * self.dt

            # Ego vehicle
            ego_vx = ego_speed
            ego_vy = 0.0
            ego_x += ego_vx * self.dt

            states.append(ActorState(
                step=s, timestamp=t, actor_id="ego", actor_type="car",
                x=ego_x, y=ego_y, vx=ego_vx, vy=ego_vy,
                speed=ego_vx, heading=0.0,
                length=ego_spec["length"], width=ego_spec["width"], is_ego=True
            ))

            # Motorcycle motion
            if s < cut_in_trigger_step:
                # Cruising parallel in adjacent gap
                mc_vx = mc_speed
                mc_vy = 0.0
                mc_heading = 0.0
            elif s < cut_in_trigger_step + 18:
                # Active cut-in across lane boundary toward y=0
                mc_vx = mc_speed * 0.95
                target_vy = -1.6 if cut_in_from_right else 1.6
                mc_vy = target_vy
                mc_heading = math.atan2(mc_vy, mc_vx)
            else:
                # Straightened out in front of ego
                mc_vx = mc_speed * 0.90
                mc_vy = 0.0
                mc_heading = 0.0

            mc_x += mc_vx * self.dt
            mc_y += mc_vy * self.dt
            # Clamp lateral to centerline if cut-in completes
            if (cut_in_from_right and mc_y < 0.0) or (not cut_in_from_right and mc_y > 0.0):
                mc_y = 0.0
                mc_vy = 0.0
                mc_heading = 0.0

            states.append(ActorState(
                step=s, timestamp=t, actor_id="mc_01", actor_type="motorcycle",
                x=mc_x, y=mc_y, vx=mc_vx, vy=mc_vy,
                speed=math.hypot(mc_vx, mc_vy), heading=mc_heading,
                length=mc_spec["length"], width=mc_spec["width"], is_ego=False
            ))

        return states, resolved_params

    def _generate_mixed_traffic_intersection(
        self, rng: np.random.RandomState, total_steps: int, params: Dict[str, Any]
    ) -> Tuple[List[ActorState], Dict[str, Any]]:
        """
        Scenario 3: Mixed-Traffic Unregulated Intersection.
        Ego travels East along x-axis. An autorickshaw approaches from cross-street heading North.
        A pedestrian walks along intersection corner.
        """
        ego_speed = params.get("ego_speed", rng.uniform(7.5, 10.5))
        auto_speed = params.get("auto_speed", rng.uniform(6.0, 9.0))
        conflict_time = rng.uniform(2.5, 4.0)

        resolved_params = {
            "ego_speed": float(ego_speed),
            "auto_speed": float(auto_speed),
            "conflict_time": float(conflict_time)
        }

        states = []
        ego_spec = ACTOR_SPECS["car"]
        auto_spec = ACTOR_SPECS["autorickshaw"]
        ped_spec = ACTOR_SPECS["pedestrian"]

        ego_x = -ego_speed * conflict_time
        ego_y = 0.0
        auto_x = 0.0
        auto_y = -auto_speed * (conflict_time + rng.uniform(-0.4, 0.4))
        ped_x = 4.0
        ped_y = 6.0

        for s in range(total_steps):
            t = s * self.dt

            # Ego
            ego_vx = ego_speed
            ego_vy = 0.0
            ego_x += ego_vx * self.dt
            states.append(ActorState(
                step=s, timestamp=t, actor_id="ego", actor_type="car",
                x=ego_x, y=ego_y, vx=ego_vx, vy=ego_vy,
                speed=ego_vx, heading=0.0,
                length=ego_spec["length"], width=ego_spec["width"], is_ego=True
            ))

            # Autorickshaw crossing perpendicular
            auto_vx = 0.0
            auto_vy = auto_speed
            auto_x += auto_vx * self.dt
            auto_y += auto_vy * self.dt
            states.append(ActorState(
                step=s, timestamp=t, actor_id="auto_01", actor_type="autorickshaw",
                x=auto_x, y=auto_y, vx=auto_vx, vy=auto_vy,
                speed=auto_speed, heading=math.pi / 2,
                length=auto_spec["length"], width=auto_spec["width"], is_ego=False
            ))

            # Corner pedestrian
            ped_vx = 0.4 * math.cos(0.3 * t)
            ped_vy = -0.5
            ped_x += ped_vx * self.dt
            ped_y += ped_vy * self.dt
            states.append(ActorState(
                step=s, timestamp=t, actor_id="ped_corner", actor_type="pedestrian",
                x=ped_x, y=ped_y, vx=ped_vx, vy=ped_vy,
                speed=math.hypot(ped_vx, ped_vy), heading=-math.pi / 2,
                length=ped_spec["length"], width=ped_spec["width"], is_ego=False
            ))

        return states, resolved_params

    def _generate_unregulated_overtaking(
        self, rng: np.random.RandomState, total_steps: int, params: Dict[str, Any]
    ) -> Tuple[List[ActorState], Dict[str, Any]]:
        """
        Scenario 4: Unregulated Overtaking & Slow Moving Vehicle.
        A slow autorickshaw/tempo moves in front, while an oncoming vehicle approaches in opposite lane.
        """
        ego_speed = params.get("ego_speed", rng.uniform(9.0, 12.0))
        slow_speed = params.get("slow_speed", rng.uniform(4.0, 6.0))
        oncoming_speed = params.get("oncoming_speed", rng.uniform(8.0, 11.0))

        resolved_params = {
            "ego_speed": float(ego_speed),
            "slow_speed": float(slow_speed),
            "oncoming_speed": float(oncoming_speed)
        }

        states = []
        ego_spec = ACTOR_SPECS["car"]
        slow_spec = ACTOR_SPECS["autorickshaw"]
        oncoming_spec = ACTOR_SPECS["car"]

        ego_x = 0.0
        slow_x = 22.0
        oncoming_x = 80.0

        for s in range(total_steps):
            t = s * self.dt

            # Ego
            ego_vx = ego_speed
            ego_x += ego_vx * self.dt
            states.append(ActorState(
                step=s, timestamp=t, actor_id="ego", actor_type="car",
                x=ego_x, y=0.0, vx=ego_vx, vy=0.0,
                speed=ego_vx, heading=0.0,
                length=ego_spec["length"], width=ego_spec["width"], is_ego=True
            ))

            # Slow lead vehicle in same lane
            slow_vx = slow_speed
            slow_x += slow_vx * self.dt
            states.append(ActorState(
                step=s, timestamp=t, actor_id="lead_slow", actor_type="autorickshaw",
                x=slow_x, y=0.0, vx=slow_vx, vy=0.0,
                speed=slow_vx, heading=0.0,
                length=slow_spec["length"], width=slow_spec["width"], is_ego=False
            ))

            # Oncoming vehicle in adjacent opposing lane (y = 3.5m)
            oncoming_vx = -oncoming_speed
            oncoming_x += oncoming_vx * self.dt
            states.append(ActorState(
                step=s, timestamp=t, actor_id="oncoming_01", actor_type="car",
                x=oncoming_x, y=3.5, vx=oncoming_vx, vy=0.0,
                speed=oncoming_speed, heading=math.pi,
                length=oncoming_spec["length"], width=oncoming_spec["width"], is_ego=False
            ))

        return states, resolved_params

    def generate_batch_dataset(
        self,
        num_episodes_per_scenario: int = 40,
        scenarios: Optional[List[str]] = None,
        base_seed: int = 1000
    ) -> List[ScenarioEpisode]:
        """
        Generate a comprehensive batch of reproducible episodes across all scenarios.
        """
        if scenarios is None:
            scenarios = [
                "pedestrian_crossing",
                "motorcycle_cut_in",
                "mixed_traffic_intersection",
                "unregulated_overtaking"
            ]

        episodes = []
        current_seed = base_seed

        for sc in scenarios:
            for i in range(num_episodes_per_scenario):
                ep = self.generate_episode(
                    scenario_type=sc,
                    seed=current_seed,
                    duration_seconds=6.0
                )
                episodes.append(ep)
                current_seed += 1

        return episodes

    @staticmethod
    def episodes_to_dataframe(episodes: List[ScenarioEpisode]) -> pd.DataFrame:
        """
        Convert list of ScenarioEpisodes to a structured tabular pandas DataFrame.
        """
        records = []
        for ep in episodes:
            for st in ep.states:
                rec = asdict(st)
                rec["episode_id"] = ep.episode_id
                rec["scenario_type"] = ep.scenario_type
                rec["random_seed"] = ep.random_seed
                rec["is_synthetic"] = ep.is_synthetic
                rec["data_label"] = ep.data_label
                records.append(rec)
        return pd.DataFrame(records)
