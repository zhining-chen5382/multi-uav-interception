"""Independent 1v1 point-defense environment for the formal PPO protocol.

Physical dynamics and Direct Attacker guidance are shared with the historical
3A1C implementation. Observation scaling and acceleration semantics must be
explicitly configured and confirmed before this environment can be constructed.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np

from .attacker_policy import DirectAttackerPolicy
from .dynamics import MotionLimits, integrate, scale_action
from .geometry import segment_distances
from .near_shaping import motion_attributed_components


OBSERVATION_FIELDS = (
    "own_vx", "own_vy", "own_vz", "own_ax", "own_ay", "own_az",
    "relative_x", "relative_y", "relative_z", "relative_vx", "relative_vy", "relative_vz",
    "attacker_base_x", "attacker_base_y", "attacker_base_z", "remaining_time_fraction",
)
REWARD_FIELDS = ("absolute_distance_cost", "survival", "capture", "attack_failure")


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must contain real finite values")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must contain real finite values") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("{} must be finite with shape {}".format(name, shape))
    return result.copy()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(name + " must be finite and positive")
    return result


class InterceptionEnv:
    """Two aircraft, one scalar reward, and a shared normalized 16D input.

    A timeout is a true terminal outcome under the formal protocol. The
    environment never truncates episodes for a trainer's rollout boundary.
    """

    observation_shape = (16,)
    action_shape = (3,)
    num_pursuers = 1
    dimension = 3

    def __init__(self, config, seed=None):
        if not isinstance(config, dict):
            raise ValueError("InterceptionEnv configuration must be a mapping")
        self.config = deepcopy(config)
        normalization = self.config.get("normalization", {})
        if normalization.get("confirmed") is not True:
            raise ValueError("Formal 16D normalization scales and acceleration_kind must be explicitly confirmed")
        self.scales = _array(normalization.get("scales"), (16,), "normalization.scales")
        if np.any(self.scales <= 0):
            raise ValueError("normalization.scales must contain 16 positive divisors")
        self.acceleration_kind = normalization.get("acceleration_kind")
        if self.acceleration_kind not in ("applied", "realized"):
            raise ValueError("normalization.acceleration_kind must be applied or realized")
        environment = self.config["environment"]
        if (environment.get("dimension", 3), environment.get("num_pursuers", 1),
                environment.get("num_evaders", 1)) != (3, 1, 1):
            raise ValueError("InterceptionEnv requires one pursuer and one attacker in 3D")
        self.dt = _positive(environment["dt"], "environment.dt")
        self.max_episode_steps = _integer(environment["max_episode_steps"], "max_episode_steps", 1)
        self.bounds_low = _array(environment["bounds"]["low"], (3,), "bounds.low")
        self.bounds_high = _array(environment["bounds"]["high"], (3,), "bounds.high")
        if np.any(self.bounds_low >= self.bounds_high):
            raise ValueError("bounds.low must be below bounds.high")
        mission = self.config["mission"]
        self.base_position = _array(mission["base_position"], (3,), "base_position")
        self.attack_radius = _positive(mission["attack_radius"], "attack_radius")
        self.capture_radius = _positive(self.config["capture"]["radius"], "capture.radius")
        self.pursuer_limits = MotionLimits.from_config(self.config["pursuer"])
        self.attacker_limits = MotionLimits.from_config(self.config["attacker"])
        policy = self.config["attacker"]["policy"]
        if policy.get("mode") != "direct":
            raise ValueError("Formal interception requires the existing Direct Attacker policy")
        self.attacker_policy = DirectAttackerPolicy(base_position=self.base_position,
            attack_speed=policy["attack_speed"], velocity_gain=policy["velocity_gain"],
            distance_epsilon=policy["distance_epsilon"])
        initialization = self.config["initialization"]
        self.pursuer_initial_position = self._single_initial(initialization, "pursuer_position", "pursuer_positions")
        self.pursuer_initial_velocity = self._single_initial(initialization, "pursuer_velocity", "pursuer_velocities")
        self.attacker_initial_position = _array(initialization["attacker_position"], (3,), "attacker_position")
        self.attacker_initial_velocity = _array(initialization["attacker_velocity"], (3,), "attacker_velocity")
        relative = self.attacker_initial_position - self.base_position
        if relative[0] <= 0 or relative[1] != 0:
            raise ValueError("Configured zero-degree attacker position must lie along the base's positive X direction")
        angles = initialization.get("angle_range_degrees", initialization.get("attacker_angle_range_degrees"))
        self.angle_range = _array(angles, (2,), "initialization.angle_range_degrees")
        if self.angle_range[0] >= self.angle_range[1]:
            raise ValueError("angle_range_degrees must have increasing endpoints")
        if "angle_range_degrees" in initialization and "attacker_angle_range_degrees" in initialization:
            if not np.array_equal(self.angle_range, initialization["attacker_angle_range_degrees"]):
                raise ValueError("Conflicting angle range aliases")
        for velocity, limits in ((self.pursuer_initial_velocity, self.pursuer_limits),
                                  (self.attacker_initial_velocity, self.attacker_limits)):
            if (np.linalg.norm(velocity[:2]) > limits.max_horizontal_speed
                    or velocity[2] > limits.max_climb_speed or velocity[2] < -limits.max_descent_speed):
                raise ValueError("Initial velocity exceeds the configured dynamics limits")
        coefficients = self.config["reward"]
        if "collision_penalty" in coefficients:
            raise ValueError("The 1v1 reward configuration must omit the inter-pursuer collision penalty")
        names = ("distance_cost_coefficient", "survival_penalty_per_second", "capture_reward", "attack_failure_penalty")
        self.reward_coefficients = {name: float(coefficients[name]) for name in names}
        if not np.isfinite(list(self.reward_coefficients.values())).all():
            raise ValueError("Reward coefficients must be finite")
        if (self.reward_coefficients["distance_cost_coefficient"] < 0
                or self.reward_coefficients["survival_penalty_per_second"] > 0
                or self.reward_coefficients["capture_reward"] <= 0
                or self.reward_coefficients["attack_failure_penalty"] >= 0):
            raise ValueError("Reward coefficient signs are invalid")
        self.near_shaping = None
        self.motion_attributed_near_enabled = False
        self.reward_fields = REWARD_FIELDS
        if "best_so_far_shaping" in coefficients:
            shaping = coefficients["best_so_far_shaping"]
            if (not isinstance(shaping, dict) or not {"length_scale_m", "beta"}.issubset(shaping)
                    or set(shaping) - {"length_scale_m", "beta", "mode"}):
                raise ValueError("best_so_far_shaping requires length_scale_m and beta")
            mode = shaping.get("mode", "record_minimum")
            if mode not in ("record_minimum", "motion_attributed"):
                raise ValueError("best_so_far_shaping.mode must be record_minimum or motion_attributed")
            scale_value = shaping["length_scale_m"]
            if isinstance(scale_value, (bool, np.bool_)):
                raise ValueError("best_so_far_shaping.length_scale_m must be finite and positive")
            scale = _positive(float(_array(scale_value, (), "best_so_far_shaping.length_scale_m")),
                              "best_so_far_shaping.length_scale_m")
            if isinstance(shaping["beta"], (bool, np.bool_)):
                raise ValueError("best_so_far_shaping.beta must be finite and nonnegative")
            beta = float(_array(shaping["beta"], (), "best_so_far_shaping.beta"))
            if beta < 0:
                raise ValueError("best_so_far_shaping.beta must be finite and nonnegative")
            self.near_shaping = dict(length_scale_m=scale, beta=beta)
            self.motion_attributed_near_enabled = mode == "motion_attributed"
            if self.motion_attributed_near_enabled:
                self.near_shaping["mode"] = mode
            self.reward_fields = REWARD_FIELDS + ("near_shaping",)
        self.line_cost_coefficient = None
        if "line_cost_coefficient" in coefficients:
            value = coefficients["line_cost_coefficient"]
            if isinstance(value, (bool, np.bool_, str, bytes)):
                raise ValueError("line_cost_coefficient must be finite and nonnegative")
            coefficient = float(_array(value, (), "line_cost_coefficient"))
            if coefficient < 0:
                raise ValueError("line_cost_coefficient must be finite and nonnegative")
            self.line_cost_coefficient = coefficient
            self.reward_fields += ("line_cost",)
        self._config_fingerprint = hashlib.sha256(json.dumps(
            self.config, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
        self.rng = np.random.default_rng(self._seed(seed))
        self._has_reset = False
        self._run_observations = 0
        self._run_violation_counts = np.zeros(16, dtype=np.int64)
        self._run_max_abs_normalized = np.zeros(16, dtype=np.float64)

    @staticmethod
    def _single_initial(initialization, singular, plural):
        if singular in initialization:
            return _array(initialization[singular], (3,), singular)
        return _array(initialization[plural], (1, 3), plural)[0]

    @staticmethod
    def _seed(seed):
        return None if seed is None else _integer(seed, "seed")

    def reset(self, seed=None, angle_degrees=None):
        if seed is not None:
            self.rng = np.random.default_rng(self._seed(seed))
        angle = float(self.rng.uniform(*self.angle_range)) if angle_degrees is None else float(angle_degrees)
        if not np.isfinite(angle) or not self.angle_range[0] <= angle <= self.angle_range[1]:
            raise ValueError("angle_degrees must be finite and within the configured range")
        self.angle_degrees = angle
        theta = np.deg2rad(angle)
        relative = self.attacker_initial_position - self.base_position
        attacker_position = self.base_position + np.array([
            relative[0] * np.cos(theta) - relative[1] * np.sin(theta),
            relative[0] * np.sin(theta) + relative[1] * np.cos(theta), relative[2]])
        positions = np.vstack((self.pursuer_initial_position, attacker_position))
        if np.any(positions < self.bounds_low) or np.any(positions > self.bounds_high):
            raise ValueError("Configured initial positions lie outside the numerical workspace")
        self.states = np.concatenate((positions, np.vstack((self.pursuer_initial_velocity,
                                                          self.attacker_initial_velocity))), axis=1)
        self.accelerations = np.zeros((2, 3), dtype=np.float64)
        self.realized_accelerations = np.zeros((2, 3), dtype=np.float64)
        self.step_count = 0
        self._done = self._captured = self._enemy_attack_success = self._timeout = False
        self._boundary_clip_counts = np.zeros(2, dtype=np.int64)
        self._boundary_clipped = np.zeros(2, dtype=bool)
        self._episode_observations = 0
        self._episode_violation_counts = np.zeros(16, dtype=np.int64)
        self._episode_max_abs_normalized = np.zeros(16, dtype=np.float64)
        self._episode_reward_components = {name: 0.0 for name in self.reward_fields}
        self._reward_components = {name: 0.0 for name in self.reward_fields}
        self.episode_return = 0.0
        self._has_reset = True
        self._episode_min_distance = self._distance()
        if self.motion_attributed_near_enabled:
            self._motion_near_diagnostics = dict(uncorrected=0.0, removed=0.0,
                                                counterfactual_distance=self._episode_min_distance)
            self._episode_motion_near_diagnostics = dict(uncorrected=0.0, removed=0.0)
        observation = self._observe()
        return observation, self._info()

    def raw_observation(self):
        """Return the physical 16D vector without incrementing diagnostics.

        The formal protocol selects ``applied``. Its acceleration field reads
        the same persistent array that step() passes to the next jerk limit.
        Realized velocity-difference acceleration remains a diagnostic quantity.
        """
        if not self._has_reset:
            raise RuntimeError("Call reset() before reading an observation")
        acceleration = self.accelerations if self.acceleration_kind == "applied" else self.realized_accelerations
        return np.concatenate((self.states[0, 3:], acceleration[0],
                               self.states[1, :3] - self.states[0, :3],
                               self.states[1, 3:] - self.states[0, 3:],
                               self.states[1, :3] - self.base_position,
                               [max(0.0, 1.0 - self.step_count / self.max_episode_steps)]))

    def _observe(self):
        raw = self.raw_observation()
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = raw / self.scales
        if not np.isfinite(normalized).all():
            raise FloatingPointError("Non-finite normalized observation")
        outside = np.abs(normalized) > 1.0
        self._run_observations += 1
        self._episode_observations += 1
        self._run_violation_counts += outside
        self._episode_violation_counts += outside
        self._run_max_abs_normalized = np.maximum(self._run_max_abs_normalized, np.abs(normalized))
        self._episode_max_abs_normalized = np.maximum(self._episode_max_abs_normalized, np.abs(normalized))
        indices = np.flatnonzero(outside)
        self._normalization_violations = dict(step=self.step_count, angle_degrees=self.angle_degrees,
            field_indices=indices.tolist(), fields=[OBSERVATION_FIELDS[i] for i in indices],
            raw_values=raw[indices].tolist(), normalized_values=normalized[indices].tolist())
        result = normalized.astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("Normalized observation exceeds float32 range")
        return result

    def _distance(self):
        return float(np.linalg.norm(self.states[1, :3] - self.states[0, :3]))

    def _line_segment_distance(self):
        return float(segment_distances(self.states[0, :3], self.states[1, :3], self.base_position))

    @property
    def best_distance_so_far(self):
        """Episode history used by the record-minimum reward and diagnostics."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before reading the episode distance history")
        return self._episode_min_distance

    def step(self, action):
        if not self._has_reset or self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        action = _array(action, (3,), "action")
        previous_best_distance = self.best_distance_so_far
        previous_pursuer_position = self.states[0, :3].copy() if self.motion_attributed_near_enabled else None
        previous_velocity = self.states[:, 3:].copy()
        pursuer, applied_pursuer = integrate(self.states[0], scale_action(action, self.pursuer_limits),
            self.accelerations[0], self.dt, self.pursuer_limits)
        attacker, applied_attacker = integrate(self.states[1], self.attacker_policy(self.states[1]),
            self.accelerations[1], self.dt, self.attacker_limits)
        self.states = np.vstack((pursuer, attacker))
        self.accelerations = np.vstack((applied_pursuer, applied_attacker))
        self.realized_accelerations = (self.states[:, 3:] - previous_velocity) / self.dt
        self._boundary_clipped = np.any((self.states[:, :3] < self.bounds_low)
                                       | (self.states[:, :3] > self.bounds_high), axis=1)
        self._boundary_clip_counts += self._boundary_clipped
        self.states[:, :3] = np.clip(self.states[:, :3], self.bounds_low, self.bounds_high)
        self.step_count += 1
        distance = self._distance()
        self._episode_min_distance = min(self._episode_min_distance, distance)
        self._captured = bool(distance < self.capture_radius)
        self._enemy_attack_success = bool(not self._captured and np.linalg.norm(
            self.states[1, :3] - self.base_position) < self.attack_radius)
        self._timeout = bool(not (self._captured or self._enemy_attack_success)
                             and self.step_count >= self.max_episode_steps)
        self._done = self._captured or self._enemy_attack_success or self._timeout
        coefficients = self.reward_coefficients
        self._reward_components = dict(
            absolute_distance_cost=-coefficients["distance_cost_coefficient"] * distance * self.dt,
            survival=coefficients["survival_penalty_per_second"] * self.dt,
            capture=coefficients["capture_reward"] if self._captured else 0.0,
            attack_failure=coefficients["attack_failure_penalty"] if self._enemy_attack_success else 0.0)
        if self.near_shaping is not None:
            scale = self.near_shaping["length_scale_m"]
            if self.motion_attributed_near_enabled:
                counterfactual_distance = float(np.linalg.norm(self.states[1, :3] - previous_pursuer_position))
                parts = motion_attributed_components(previous_best_distance, self.best_distance_so_far,
                    counterfactual_distance, scale, self.near_shaping["beta"])
                self._reward_components["near_shaping"] = float(parts["active_near"])
                self._motion_near_diagnostics = dict(uncorrected=float(parts["original_near"]),
                    removed=float(parts["removed_near"]), counterfactual_distance=counterfactual_distance)
                self._episode_motion_near_diagnostics["uncorrected"] += float(parts["original_near"])
                self._episode_motion_near_diagnostics["removed"] += float(parts["removed_near"])
            else:
                self._reward_components["near_shaping"] = float(self.near_shaping["beta"] * (
                    np.exp(-self.best_distance_so_far / scale) - np.exp(-previous_best_distance / scale)))
        if self.line_cost_coefficient is not None:
            self._reward_components["line_cost"] = -self.line_cost_coefficient * self._line_segment_distance() * self.dt
        reward = float(sum(self._reward_components.values()))
        if not np.isfinite(reward):
            raise FloatingPointError("Non-finite episode reward")
        self.episode_return += reward
        for name, value in self._reward_components.items():
            self._episode_reward_components[name] += value
        observation = self._observe()
        return observation, reward, self._done, False, self._info()

    def _info(self):
        outcome = ("Interception Success" if self._captured else "Enemy Attack Success"
                   if self._enemy_attack_success else "Timeout" if self._timeout else "Ongoing")
        info = dict(states=self.states.copy(), raw_observation=self.raw_observation(),
            angle_degrees=self.angle_degrees, initial_angle_degrees=self.angle_degrees,
            step_count=self.step_count, min_distance=self._distance(),
            episode_min_distance=self._episode_min_distance,
            distance_to_base=float(np.linalg.norm(self.states[1, :3] - self.base_position)),
            captured=self._captured, interception_success=self._captured,
            enemy_attack_success=self._enemy_attack_success, attacker_reached_base=self._enemy_attack_success,
            timeout=self._timeout, terminated=self._done, truncated=False, outcome=outcome,
            capture_time=float(self.step_count * self.dt) if self._captured else None,
            episode_return=self.episode_return, reward_components=deepcopy(self._reward_components),
            episode_reward_components=deepcopy(self._episode_reward_components),
            applied_accelerations=self.accelerations.copy(), realized_accelerations=self.realized_accelerations.copy(),
            boundary_clip_count=int(self._boundary_clip_counts.sum()),
            boundary_clip_counts=self._boundary_clip_counts.copy(), boundary_clipped=self._boundary_clipped.copy(),
            normalization_violations=deepcopy(self._normalization_violations),
            normalization_violation_mask=np.abs(self.raw_observation() / self.scales) > 1.0,
            normalization_episode_observations=self._episode_observations,
            normalization_run_observations=self._run_observations,
            normalization_episode_violation_counts=self._episode_violation_counts.copy(),
            normalization_run_violation_counts=self._run_violation_counts.copy(),
            normalization_episode_max_abs=self._episode_max_abs_normalized.copy(),
            normalization_run_max_abs=self._run_max_abs_normalized.copy())
        if self.near_shaping is not None:
            info["best_distance_so_far"] = self.best_distance_so_far
        if self.motion_attributed_near_enabled:
            info.update(near_shaping_uncorrected=self._motion_near_diagnostics["uncorrected"],
                near_shaping_removed=self._motion_near_diagnostics["removed"],
                episode_near_shaping_uncorrected=self._episode_motion_near_diagnostics["uncorrected"],
                episode_near_shaping_removed=self._episode_motion_near_diagnostics["removed"],
                near_counterfactual_distance=self._motion_near_diagnostics["counterfactual_distance"])
        if self.line_cost_coefficient is not None:
            info["line_segment_distance"] = self._line_segment_distance()
        return info

    def get_state(self):
        """Snapshot physical state, local RNG and all episode/run diagnostics."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before snapshotting the environment")
        snapshot = dict(format_version=1, config_fingerprint=self._config_fingerprint,
            states=self.states.copy(), accelerations=self.accelerations.copy(),
            realized_accelerations=self.realized_accelerations.copy(), step_count=self.step_count,
            angle_degrees=self.angle_degrees, done=self._done, captured=self._captured,
            enemy_attack_success=self._enemy_attack_success, timeout=self._timeout,
            rng_state=deepcopy(self.rng.bit_generator.state), episode_return=self.episode_return,
            episode_min_distance=self._episode_min_distance,
            reward_components=deepcopy(self._reward_components),
            episode_reward_components=deepcopy(self._episode_reward_components),
            boundary_clip_counts=self._boundary_clip_counts.copy(), boundary_clipped=self._boundary_clipped.copy(),
            run_observations=self._run_observations, episode_observations=self._episode_observations,
            run_violation_counts=self._run_violation_counts.copy(),
            episode_violation_counts=self._episode_violation_counts.copy(),
            run_max_abs_normalized=self._run_max_abs_normalized.copy(),
            episode_max_abs_normalized=self._episode_max_abs_normalized.copy(),
            normalization_violations=deepcopy(self._normalization_violations))
        if self.near_shaping is not None:
            snapshot["best_distance_so_far"] = self.best_distance_so_far
        if self.motion_attributed_near_enabled:
            snapshot["motion_near_diagnostics"] = deepcopy(self._motion_near_diagnostics)
            snapshot["episode_motion_near_diagnostics"] = deepcopy(self._episode_motion_near_diagnostics)
        return snapshot

    def set_state(self, state):
        """Restore a complete snapshot atomically, without emitting a new observation."""
        if (not isinstance(state, dict) or state.get("format_version") != 1
                or state.get("config_fingerprint") != self._config_fingerprint):
            raise ValueError("Environment snapshot version or configuration differs")
        restored = deepcopy(state)
        for name, shape in (("states", (2, 6)), ("accelerations", (2, 3)),
                            ("realized_accelerations", (2, 3)), ("boundary_clip_counts", (2,)),
                            ("boundary_clipped", (2,)), ("run_violation_counts", (16,)),
                            ("episode_violation_counts", (16,)), ("run_max_abs_normalized", (16,)),
                            ("episode_max_abs_normalized", (16,))):
            restored[name] = _array(restored[name], shape, name)
        for name in ("step_count", "run_observations", "episode_observations"):
            _integer(restored[name], name)
        if restored["step_count"] > self.max_episode_steps:
            raise ValueError("Snapshot step_count exceeds the episode limit")
        for name in ("done", "captured", "enemy_attack_success", "timeout"):
            if not isinstance(restored[name], bool):
                raise ValueError("Snapshot outcome flags must be booleans")
        outcomes = sum(restored[name] for name in ("captured", "enemy_attack_success", "timeout"))
        if outcomes > 1 or restored["done"] != bool(outcomes):
            raise ValueError("Snapshot outcome flags are inconsistent")
        for name in ("angle_degrees", "episode_return", "episode_min_distance"):
            if not np.isfinite(restored[name]):
                raise ValueError("Snapshot scalar must be finite: " + name)
        if not self.angle_range[0] <= restored["angle_degrees"] <= self.angle_range[1]:
            raise ValueError("Snapshot angle is outside the configured range")
        if self.near_shaping is not None:
            best = restored.get("best_distance_so_far")
            if best is None or isinstance(best, (bool, np.bool_)):
                raise ValueError("Snapshot best_distance_so_far is missing or inconsistent")
            best = float(_array(best, (), "best_distance_so_far"))
            if best < 0 or best != restored["episode_min_distance"]:
                raise ValueError("Snapshot best_distance_so_far is missing or inconsistent")
            current_distance = float(np.linalg.norm(restored["states"][1, :3] - restored["states"][0, :3]))
            if best > current_distance:
                raise ValueError("Snapshot best_distance_so_far exceeds the current distance")
        for name in ("reward_components", "episode_reward_components"):
            if set(restored[name]) != set(self.reward_fields):
                raise ValueError("Snapshot reward components are invalid")
            if self.line_cost_coefficient is not None:
                value = restored[name]["line_cost"]
                if isinstance(value, (bool, np.bool_, str, bytes)):
                    raise ValueError("Snapshot line_cost must be finite and nonpositive")
                line_value = float(_array(value, (), "Snapshot line_cost"))
                if line_value > 0:
                    raise ValueError("Snapshot line_cost must be nonpositive")
            if not np.isfinite(list(restored[name].values())).all():
                raise ValueError("Snapshot reward components are invalid")
        for name in ("boundary_clip_counts", "run_violation_counts", "episode_violation_counts"):
            if np.any(restored[name] < 0) or np.any(restored[name] != np.floor(restored[name])):
                raise ValueError("Snapshot diagnostic counts must be nonnegative integers")
            restored[name] = restored[name].astype(np.int64)
        if self.motion_attributed_near_enabled:
            for name, fields in (("motion_near_diagnostics", {"uncorrected", "removed", "counterfactual_distance"}),
                                 ("episode_motion_near_diagnostics", {"uncorrected", "removed"})):
                if not isinstance(restored.get(name), dict) or set(restored[name]) != fields:
                    raise ValueError("Snapshot motion near diagnostics are missing or invalid")
                for field in fields:
                    value = restored[name][field]
                    if isinstance(value, (bool, np.bool_, str, bytes)):
                        raise ValueError("Snapshot motion near diagnostics must be finite and nonnegative")
                    restored[name][field] = float(_array(value, (), "Snapshot motion near diagnostics"))
                    if restored[name][field] < 0:
                        raise ValueError("Snapshot motion near diagnostics must be nonnegative")
            for name, reward_name in (("motion_near_diagnostics", "reward_components"),
                                     ("episode_motion_near_diagnostics", "episode_reward_components")):
                diagnostic = restored[name]
                active = restored[reward_name]["near_shaping"]
                if (active < 0 or diagnostic["removed"] > diagnostic["uncorrected"]
                        or diagnostic["uncorrected"] > self.near_shaping["beta"] + 1e-8
                        or not np.isclose(diagnostic["uncorrected"] - diagnostic["removed"], active,
                                          rtol=0, atol=1e-8)):
                    raise ValueError("Snapshot motion near reward accounting is inconsistent")
            if any(restored["motion_near_diagnostics"][field]
                   > restored["episode_motion_near_diagnostics"][field] + 1e-8 for field in ("uncorrected", "removed")):
                raise ValueError("Snapshot motion near episode totals are inconsistent")
        restored["boundary_clipped"] = restored["boundary_clipped"].astype(bool)
        candidate_rng = np.random.default_rng()
        candidate_rng.bit_generator.state = deepcopy(restored["rng_state"])
        assignments = dict(states="states", accelerations="accelerations", realized_accelerations="realized_accelerations",
            step_count="step_count", angle_degrees="angle_degrees", _done="done", _captured="captured",
            _enemy_attack_success="enemy_attack_success", _timeout="timeout", episode_return="episode_return",
            _episode_min_distance="episode_min_distance", _reward_components="reward_components",
            _episode_reward_components="episode_reward_components", _boundary_clip_counts="boundary_clip_counts",
            _boundary_clipped="boundary_clipped", _run_observations="run_observations",
            _episode_observations="episode_observations", _run_violation_counts="run_violation_counts",
            _episode_violation_counts="episode_violation_counts", _run_max_abs_normalized="run_max_abs_normalized",
            _episode_max_abs_normalized="episode_max_abs_normalized", _normalization_violations="normalization_violations")
        if self.motion_attributed_near_enabled:
            assignments.update(_motion_near_diagnostics="motion_near_diagnostics",
                               _episode_motion_near_diagnostics="episode_motion_near_diagnostics")
        for target, source in assignments.items():
            setattr(self, target, restored[source])
        self.rng = candidate_rng
        self._has_reset = True
