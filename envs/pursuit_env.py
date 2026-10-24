"""Minimal NumPy environment for the 3A1C Baseline v1.

This uses a five-value step interface, but is not a Gymnasium Env. All three
pursuers share a reward. States are ordered [pursuer_0, pursuer_1, pursuer_2,
evader], each as [x, y, z, vx, vy, vz]. Accelerations are kept separately for
jerk limiting and are not appended to the specified 21/24-dimensional inputs.
"""

from pathlib import Path

import numpy as np
import yaml

from .dynamics import MotionLimits, integrate, scale_action
from .evader_policy import RepulsiveEvaderPolicy


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
        self.step_count = 0
        self._has_reset = False
        self._done = False

    def _configure(self):
        config = self.config
        environment = config["environment"]
        self.dimension = environment["dimension"]
        self.num_pursuers = environment["num_pursuers"]
        if (self.dimension, self.num_pursuers, environment["num_evaders"]) != (3, 3, 1):
            raise ValueError("Baseline v1 requires 3D, three pursuers and one evader")
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
        self.evader_limits = MotionLimits.from_config(config["evader"])
        self.evader_policy = RepulsiveEvaderPolicy(
            self.evader_limits, config["evader"]["policy"]["distance_epsilon"]
        )
        self.capture_radius = _positive(config["capture"]["radius"], "capture.radius")
        self.safety_distance = _positive(
            config["safety"]["pursuer_safety_distance"], "safety.pursuer_safety_distance"
        )
        self.reward_coefficients = {}
        for name in ("capture_reward", "progress_coefficient", "time_penalty", "collision_penalty"):
            value = float(config["reward"][name])
            if not np.isfinite(value):
                raise ValueError("reward.{} must be finite".format(name))
            self.reward_coefficients[name] = value

        initialization = config["initialization"]
        positions = np.vstack((
            _array(initialization["pursuer_positions"], (3, 3), "pursuer_positions"),
            _array(initialization["evader_position"], (3,), "evader_position"),
        ))
        velocities = np.vstack((
            _array(initialization["pursuer_velocities"], (3, 3), "pursuer_velocities"),
            _array(initialization["evader_velocity"], (3,), "evader_velocity"),
        ))
        if np.any(positions < self.bounds_low) or np.any(positions > self.bounds_high):
            raise ValueError("Initial positions must be within environment bounds")
        for velocity, limits in zip(velocities, [self.pursuer_limits] * 3 + [self.evader_limits]):
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
        self.step_count = 0
        self._has_reset = True
        self._done = False
        return self._observations(), self._info()

    def sample_actions(self):
        """Sample independent uniform normalized actions using this env's RNG."""
        return self.rng.uniform(-1.0, 1.0, size=self.action_shape)

    def get_state(self):
        """Return a copy of [p0_state, p1_state, p2_state, evader_state]."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before reading environment state")
        return self.states.reshape(-1).copy()

    def step(self, actions):
        """Advance all four aircraft synchronously from the same prior state."""
        if not self._has_reset or self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        actions = _array(actions, self.action_shape, "actions")
        previous_min_distance = self._min_distance()
        pursuer_command = scale_action(actions, self.pursuer_limits)
        evader_command = self.evader_policy(self.states[-1], self.states[:-1])
        pursuer_states, pursuer_accelerations = integrate(
            self.states[:-1], pursuer_command, self.accelerations[:-1],
            self.dt, self.pursuer_limits,
        )
        evader_state, evader_acceleration = integrate(
            self.states[-1], evader_command, self.accelerations[-1],
            self.dt, self.evader_limits,
        )
        self.states = np.vstack((pursuer_states, evader_state))
        self.accelerations = np.vstack((pursuer_accelerations, evader_acceleration))
        # Minimal bounded-space handling. Velocity is retained; no wall bounce,
        # boundary reward or extra termination is introduced.
        self.states[:, :3] = np.clip(self.states[:, :3], self.bounds_low, self.bounds_high)
        self.step_count += 1

        info = self._info()
        terminated = info["captured"]
        truncated = self.step_count >= self.max_episode_steps and not terminated
        coefficients = self.reward_coefficients
        components = {
            "capture": coefficients["capture_reward"] if terminated else 0.0,
            "progress": coefficients["progress_coefficient"] * (
                previous_min_distance - info["min_distance"]
            ),
            "time": coefficients["time_penalty"],
            # A single team penalty per step whenever any pair is unsafe.
            "collision": coefficients["collision_penalty"] if info["collision_pairs"] else 0.0,
        }
        rewards = np.full(self.num_pursuers, sum(components.values()), dtype=np.float64)
        info["reward_components"] = components
        self._done = terminated or truncated
        return self._observations(), rewards, terminated, truncated, info

    def _observations(self):
        """Order: own v, relative evader p/v, teammate p, teammate v.

        Teammates are always sorted by their fixed agent indices, excluding
        the observing agent. Teammate positions precede all teammate velocities.
        Values are physical relative states without normalization or noise.
        """
        observations = []
        evader = self.states[-1]
        for index, pursuer in enumerate(self.states[:-1]):
            teammates = np.delete(self.states[:-1], index, axis=0)
            observations.append(np.concatenate((
                pursuer[3:], evader[:3] - pursuer[:3], evader[3:] - pursuer[3:],
                (teammates[:, :3] - pursuer[:3]).reshape(-1),
                (teammates[:, 3:] - pursuer[3:]).reshape(-1),
            )))
        return np.asarray(observations)

    def _min_distance(self):
        return float(np.min(np.linalg.norm(self.states[:-1, :3] - self.states[-1, :3], axis=1)))

    def _info(self):
        min_distance = self._min_distance()
        collision_pairs = sum(
            np.linalg.norm(self.states[i, :3] - self.states[j, :3]) < self.safety_distance
            for i in range(self.num_pursuers) for j in range(i + 1, self.num_pursuers)
        )
        return {
            "global_state": self.get_state(),
            "min_distance": min_distance,
            "captured": min_distance < self.capture_radius,
            "step_count": self.step_count,
            "collision_pairs": int(collision_pairs),
        }
