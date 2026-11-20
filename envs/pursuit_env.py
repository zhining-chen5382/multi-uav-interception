"""NumPy point-defense environment for the 3A1C Baseline v1.

This uses a five-value step interface, but is not a Gymnasium Env. All three
pursuers share a reward. States are ordered [pursuer_0, pursuer_1, pursuer_2,
attacker], each as [x, y, z, vx, vy, vz]. Accelerations are kept separately for
jerk limiting and are not appended to the specified 21/24-dimensional inputs.
"""

from pathlib import Path

import numpy as np
import yaml

from .dynamics import MotionLimits, integrate, scale_action
from .attacker_policy import DirectAttackerPolicy, GoalDirectedAttackerPolicy
from .evader_policy import RepulsiveEvaderPolicy
from .rewards import (
    ABSOLUTE_DISTANCE_REWARD_TERMS, ACTIVE_REWARD_TERMS,
    absolute_distance_components, active_progress_components,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline_3a1c.yaml"


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError("{} must contain real, finite values".format(name))
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("{} must be finite with shape {}".format(name, shape))
    return result


def _positive(value, name):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("{} must be finite and positive".format(name))
    return result


class PursuitEnv:
    """Fixed-initialization, obstacle-free 3D environment loaded from YAML.

    reset(seed=None) -> observations, info
    step(actions) -> observations, rewards, terminated, truncated, info

    Actions have shape (3, 3) in [-1, 1]; finite out-of-range values are clipped.
    Observation shape is (3, 21), rewards shape is (3,), and get_state() returns
    the 24-dimensional centralized state. All arrays use float64.
    """

    def __init__(self, config_path=DEFAULT_CONFIG):
        self.config_path = Path(config_path).resolve()
        with self.config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a YAML mapping")
        try:
            self._configure()
        except KeyError as error:
            raise ValueError("Missing required configuration key: {}".format(error)) from error
        self.rng = np.random.default_rng()
        self.states = np.zeros_like(self._initial_states)
        self.accelerations = np.zeros((self.num_pursuers + 1, self.dimension))
        self._boundary_clip_counts = np.zeros(self.num_pursuers + 1, dtype=np.int64)
        self._boundary_clipped = np.zeros(self.num_pursuers + 1, dtype=bool)
        self._episode_reward_components = {name: 0.0 for name in self._reward_terms}
        self.step_count = 0
        self._has_reset = False
        self._done = False

    def _configure(self):
        config = self.config
        environment = config["environment"]
        self.dimension = environment["dimension"]
        self.num_pursuers = environment["num_pursuers"]
        if (self.dimension, self.num_pursuers, environment["num_evaders"]) != (3, 3, 1):
            raise ValueError("Baseline v1 requires 3D, three pursuers and one attacker")
        self.dt = _positive(environment["dt"], "environment.dt")
        self.max_episode_steps = environment["max_episode_steps"]
        if (isinstance(self.max_episode_steps, bool)
                or not isinstance(self.max_episode_steps, int)
                or self.max_episode_steps <= 0):
            raise ValueError("environment.max_episode_steps must be a positive integer")
        self.bounds_low = _array(environment["bounds"]["low"], (3,), "bounds.low")
        self.bounds_high = _array(environment["bounds"]["high"], (3,), "bounds.high")
        if np.any(self.bounds_low >= self.bounds_high):
            raise ValueError("Each bounds.low component must be less than bounds.high")
        self.pursuer_limits = MotionLimits.from_config(config["pursuer"])
        self.mission_enabled = "mission" in config
        if "attacker" in config and not self.mission_enabled:
            raise ValueError("attacker configuration requires a mission section")
        if self.mission_enabled:
            mission = config["mission"]
            self.base_position = _array(mission["base_position"], (3,), "mission.base_position").copy()
            if np.any(self.base_position < self.bounds_low) or np.any(self.base_position > self.bounds_high):
                raise ValueError("mission.base_position must be within numerical workspace")
            self.detection_radius = _positive(mission["detection_radius"], "mission.detection_radius")
            self.attack_radius = _positive(mission["attack_radius"], "mission.attack_radius")
            if self.attack_radius >= self.detection_radius:
                raise ValueError("mission.attack_radius must be less than detection_radius")
        else:
            # Historical snapshots are explicitly identified by their old schema.
            self.base_position = self.detection_radius = self.attack_radius = None
        aircraft_name = "attacker" if self.mission_enabled else "evader"
        self.attacker_limits = MotionLimits.from_config(config[aircraft_name])
        policy_config = config[aircraft_name]["policy"]
        boundary_keys = ("boundary_margin_horizontal", "boundary_margin_vertical", "boundary_strength")
        boundary_options = {key: policy_config[key] for key in boundary_keys if key in policy_config}
        if boundary_options:
            if len(boundary_options) != len(boundary_keys):
                raise ValueError("{}.policy must provide all three boundary steering parameters".format(aircraft_name))
            boundary_options.update(bounds_low=self.bounds_low, bounds_high=self.bounds_high)
        if self.mission_enabled:
            mode = policy_config.get("mode")
            if mode == "direct":
                if boundary_options:
                    raise ValueError("Direct attacker does not use soft boundary steering")
                if "goal_weight" in policy_config or policy_config.get("avoidance_weight", 0) != 0:
                    raise ValueError("Direct attacker does not use goal/avoidance force weights")
                self.attacker_policy = DirectAttackerPolicy(
                    base_position=self.base_position, attack_speed=policy_config["attack_speed"],
                    velocity_gain=policy_config["velocity_gain"],
                    distance_epsilon=policy_config["distance_epsilon"],
                )
            elif mode is None:
                # Missing mode identifies historical goal-attraction snapshots;
                # it never silently upgrades their stored experimental behavior.
                self.attacker_policy = GoalDirectedAttackerPolicy(
                    self.attacker_limits, policy_config["distance_epsilon"],
                    base_position=self.base_position, goal_weight=policy_config["goal_weight"],
                    avoidance_weight=policy_config["avoidance_weight"], **boundary_options
                )
            else:
                raise ValueError("Unsupported attacker.policy.mode: {!r}".format(mode))
        else:
            self.attacker_policy = RepulsiveEvaderPolicy(
                self.attacker_limits, policy_config["distance_epsilon"], **boundary_options
            )
        # Keep old attribute readers/checkpoint tooling compatible without
        # retaining pure evasion as the default point-defense behavior.
        self.evader_limits = self.attacker_limits
        self.evader_policy = self.attacker_policy
        self.capture_radius = _positive(config["capture"]["radius"], "capture.radius")
        self.safety_distance = _positive(
            config["safety"]["pursuer_safety_distance"], "safety.pursuer_safety_distance"
        )
        reward_config = config["reward"]
        active_names = ("active_progress_coefficient", "survival_penalty_per_second")
        distance_names = ("distance_cost_coefficient", "survival_penalty_per_second")
        legacy_names = ("progress_coefficient", "time_penalty")
        self.active_progress_enabled = "active_progress_coefficient" in reward_config
        self.absolute_distance_enabled = "distance_cost_coefficient" in reward_config
        if self.absolute_distance_enabled:
            if self.active_progress_enabled or any(name in reward_config for name in legacy_names):
                raise ValueError("Cannot mix absolute distance reward with active or legacy progress/time keys")
            if not self.mission_enabled:
                raise ValueError("Absolute distance reward requires a mission/base position")
            reward_names = ["capture_reward", *distance_names, "collision_penalty"]
            self._reward_terms = ABSOLUTE_DISTANCE_REWARD_TERMS
        elif self.active_progress_enabled:
            if not self.mission_enabled:
                raise ValueError("Active progress reward requires a mission/base position")
            if any(name in reward_config for name in legacy_names):
                raise ValueError("Cannot mix active progress reward with legacy progress/time keys")
            reward_names = ["capture_reward", *active_names, "collision_penalty"]
            self._reward_terms = ACTIVE_REWARD_TERMS
        else:
            if "survival_penalty_per_second" in reward_config:
                raise ValueError("survival_penalty_per_second requires an active or absolute distance reward schema")
            # Stored historical YAMLs retain their original reward. Schemas
            # are exclusive: historical shaping is never silently combined.
            reward_names = ["capture_reward", *legacy_names, "collision_penalty"]
            self._reward_terms = ("capture", "progress", "time", "collision")
            if self.mission_enabled:
                self._reward_terms += ("attack_failure",)
        self.reward_coefficients = {}
        if self.mission_enabled:
            reward_names.append("attack_failure_penalty")
        for name in reward_names:
            value = float(reward_config[name])
            if not np.isfinite(value):
                raise ValueError("reward.{} must be finite".format(name))
            self.reward_coefficients[name] = value
        if self.mission_enabled and self.reward_coefficients["attack_failure_penalty"] >= 0:
            raise ValueError("reward.attack_failure_penalty must be negative")
        if self.active_progress_enabled:
            if self.reward_coefficients["active_progress_coefficient"] < 0:
                raise ValueError("reward.active_progress_coefficient must be nonnegative")
        if self.absolute_distance_enabled:
            if self.reward_coefficients["distance_cost_coefficient"] <= 0:
                raise ValueError("reward.distance_cost_coefficient must be positive")
        if self.active_progress_enabled or self.absolute_distance_enabled:
            if self.reward_coefficients["survival_penalty_per_second"] > 0:
                raise ValueError("reward.survival_penalty_per_second must be nonpositive")

        initialization = config["initialization"]
        positions = np.vstack((
            _array(initialization["pursuer_positions"], (3, 3), "pursuer_positions"),
            _array(initialization[aircraft_name + "_position"], (3,), aircraft_name + "_position"),
        ))
        velocities = np.vstack((
            _array(initialization["pursuer_velocities"], (3, 3), "pursuer_velocities"),
            _array(initialization[aircraft_name + "_velocity"], (3,), aircraft_name + "_velocity"),
        ))
        if np.any(positions < self.bounds_low) or np.any(positions > self.bounds_high):
            raise ValueError("Initial positions must be within environment bounds")
        for velocity, limits in zip(velocities, [self.pursuer_limits] * 3 + [self.attacker_limits]):
            if (np.linalg.norm(velocity[:2]) > limits.max_horizontal_speed
                    or velocity[2] > limits.max_climb_speed
                    or velocity[2] < -limits.max_descent_speed):
                raise ValueError("Initial velocities must satisfy configured speed limits")
        self._initial_states = np.concatenate((positions, velocities), axis=1)
        self.action_shape = (self.num_pursuers, self.dimension)
        self.observation_shape = (self.num_pursuers, 21)

    def reset(self, seed=None):
        """Restore configured states and zero acceleration; seed the action RNG.

        Initialization is fixed. Repeated reset(seed=None) continues the RNG
        stream; reset(seed=...) restarts it for reproducible random rollouts.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.states = self._initial_states.copy()
        self.accelerations = np.zeros((self.num_pursuers + 1, self.dimension))
        self._boundary_clip_counts = np.zeros(self.num_pursuers + 1, dtype=np.int64)
        self._boundary_clipped = np.zeros(self.num_pursuers + 1, dtype=bool)
        self._episode_reward_components = {name: 0.0 for name in self._reward_terms}
        self.step_count = 0
        self._has_reset = True
        self._done = False
        return self._observations(), self._info()

    def sample_actions(self):
        """Sample independent uniform normalized actions using this env's RNG."""
        return self.rng.uniform(-1.0, 1.0, size=self.action_shape)

    def get_state(self):
        """Return a copy of [p0_state, p1_state, p2_state, attacker_state]."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before reading environment state")
        return self.states.reshape(-1).copy()

    def step(self, actions):
        """Advance all four aircraft synchronously from the same prior state."""
        if not self._has_reset or self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        actions = _array(actions, self.action_shape, "actions")
        previous_min_distance = self._min_distance()
        previous_base_distance = (
            float(np.linalg.norm(self.states[-1, :3] - self.base_position))
            if self.active_progress_enabled else None
        )
        pursuer_command = scale_action(actions, self.pursuer_limits)
        attacker_command = self.attacker_policy(self.states[-1], self.states[:-1])
        pursuer_states, pursuer_accelerations = integrate(
            self.states[:-1], pursuer_command, self.accelerations[:-1],
            self.dt, self.pursuer_limits,
        )
        attacker_state, attacker_acceleration = integrate(
            self.states[-1], attacker_command, self.accelerations[-1],
            self.dt, self.attacker_limits,
        )
        self.states = np.vstack((pursuer_states, attacker_state))
        self.accelerations = np.vstack((pursuer_accelerations, attacker_acceleration))
        # Count actual out-of-bounds proposals before applying the unchanged
        # numerical safety clip. One event per aircraft per step, even if more
        # than one coordinate is clipped. No pursuer steering or velocity reset.
        self._boundary_clipped = np.any(
            (self.states[:, :3] < self.bounds_low) | (self.states[:, :3] > self.bounds_high), axis=1
        )
        self._boundary_clip_counts += self._boundary_clipped
        self.states[:, :3] = np.clip(self.states[:, :3], self.bounds_low, self.bounds_high)
        self.step_count += 1

        info = self._info()
        terminated = info["interception_success"] or info["attacker_reached_base"]
        truncated = self.step_count >= self.max_episode_steps and not terminated
        coefficients = self.reward_coefficients
        if self.absolute_distance_enabled:
            components = absolute_distance_components(
                info["min_distance"], self.dt, coefficients,
                captured=info["captured"], attacker_reached_base=info["attacker_reached_base"],
                collision_pairs=info["collision_pairs"],
            )
        elif self.active_progress_enabled:
            components = active_progress_components(
                previous_min_distance, info["min_distance"],
                previous_base_distance, info["distance_to_base"], self.dt, coefficients,
                captured=info["captured"], attacker_reached_base=info["attacker_reached_base"],
                collision_pairs=info["collision_pairs"],
            )
        else:
            components = {
                "capture": coefficients["capture_reward"] if info["captured"] else 0.0,
                "progress": coefficients["progress_coefficient"] * (
                    previous_min_distance - info["min_distance"]
                ),
                "time": coefficients["time_penalty"],
                # A single team penalty per step whenever any pair is unsafe.
                "collision": coefficients["collision_penalty"] if info["collision_pairs"] else 0.0,
            }
            if self.mission_enabled:
                components["attack_failure"] = (
                    coefficients["attack_failure_penalty"] if info["attacker_reached_base"] else 0.0
                )
        # Raw distance/progress diagnostics are not extra reward terms.
        team_reward = sum(components[name] for name in self._reward_terms)
        totals = {name: self._episode_reward_components[name] + components[name]
                  for name in self._reward_terms}
        if not np.isfinite(team_reward) or not all(np.isfinite(value) for value in totals.values()):
            raise ValueError("Reward computation exceeds the finite numerical range")
        rewards = np.full(self.num_pursuers, team_reward, dtype=np.float64)
        self._episode_reward_components = totals
        info["reward_components"] = components.copy()
        info["episode_reward_components"] = self._episode_reward_components.copy()
        self._done = terminated or truncated
        return self._observations(), rewards, terminated, truncated, info

    def _observations(self):
        """Order: own v, relative attacker p/v, teammate p, teammate v.

        Teammates are always sorted by their fixed agent indices, excluding
        the observing agent. Teammate positions precede all teammate velocities.
        Values are physical relative states without normalization or noise.
        """
        observations = []
        attacker = self.states[-1]
        for index, pursuer in enumerate(self.states[:-1]):
            teammates = np.delete(self.states[:-1], index, axis=0)
            observations.append(np.concatenate((
                pursuer[3:], attacker[:3] - pursuer[:3], attacker[3:] - pursuer[3:],
                (teammates[:, :3] - pursuer[:3]).reshape(-1),
                (teammates[:, 3:] - pursuer[3:]).reshape(-1),
            )))
        return np.asarray(observations)

    def _min_distance(self):
        return float(np.min(np.linalg.norm(self.states[:-1, :3] - self.states[-1, :3], axis=1)))

    def _info(self):
        min_distance = self._min_distance()
        captured = min_distance < self.capture_radius
        distance_to_base = (float(np.linalg.norm(self.states[-1, :3] - self.base_position))
                            if self.mission_enabled else None)
        in_attack_region = self.mission_enabled and distance_to_base < self.attack_radius
        # Judge the same post-update state, with interception winning ties.
        attacker_reached_base = bool(in_attack_region and not captured)
        collision_pairs = sum(
            np.linalg.norm(self.states[i, :3] - self.states[j, :3]) < self.safety_distance
            for i in range(self.num_pursuers) for j in range(i + 1, self.num_pursuers)
        )
        return {
            "global_state": self.get_state(),
            "min_distance": min_distance,
            "captured": captured,
            "interception_success": captured,
            "attacker_reached_base": attacker_reached_base,
            "enemy_attack_success": attacker_reached_base,
            "interception_failure": attacker_reached_base,
            "attacker_in_attack_region": bool(in_attack_region),
            "distance_to_base": distance_to_base,
            "mission_enabled": self.mission_enabled,
            "step_count": self.step_count,
            "collision_pairs": int(collision_pairs),
            "boundary_clip_count": int(self._boundary_clip_counts.sum()),
            "boundary_clip_counts": self._boundary_clip_counts.copy(),
            "boundary_clipped": self._boundary_clipped.copy(),
            "boundary_steering_enabled": self.attacker_policy.boundary_enabled,
            "episode_reward_components": self._episode_reward_components.copy(),
        }
