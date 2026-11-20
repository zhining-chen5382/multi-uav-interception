"""Geometry, evaluation ranking and checkpoint transfer for v09 curricula.

The original environment and MAPPO implementation remain unchanged. The
environment subclass changes only its own attacker initial position at reset.
"""

import math
import random

import numpy as np
import torch

from algorithms.mappo import MAPPOAgent
from envs import PursuitEnv


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(name + " must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " must be a finite real scalar") from error
    if not math.isfinite(result):
        raise ValueError(name + " must be a finite real scalar")
    return result


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _vector(value, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must contain three real finite coordinates")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " must contain three real finite coordinates") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + " must contain three real finite coordinates")
    return result


def arrival_position(base_environment_config, angle_degrees):
    """Rotate the original attacker around the base's vertical axis.

    The angle is an offset from the snapshotted original direction. Rotation
    preserves horizontal radius, distance to the base, and absolute height;
    neither spherical projection nor coordinate clipping is applied.
    """
    angle = _finite_number(angle_degrees, "angle_degrees")
    try:
        base = _vector(base_environment_config["mission"]["base_position"], "base_position")
        original = _vector(base_environment_config["initialization"]["attacker_position"], "attacker_position")
        low = _vector(base_environment_config["environment"]["bounds"]["low"], "bounds.low")
        high = _vector(base_environment_config["environment"]["bounds"]["high"], "bounds.high")
    except (KeyError, TypeError) as error:
        raise ValueError("Arrival-direction sampling requires a complete mission configuration") from error
    if np.any(low >= high):
        raise ValueError("Every lower workspace bound must be smaller than its upper bound")
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    relative = original[:2] - base[:2]
    result = original.copy()
    result[:2] = base[:2] + np.array([
        cosine * relative[0] - sine * relative[1],
        sine * relative[0] + cosine * relative[1],
    ])
    if not np.isfinite(result).all() or np.any(result < low) or np.any(result > high):
        raise ValueError("Rotated attacker initial position exceeds workspace bounds")
    return result


class DirectionCurriculumEnv(PursuitEnv):
    """Per-episode continuous azimuth sampling with the v08 physical task.

    ``fixed_angle_degrees`` selects one evaluation angle without drawing from
    the independent training sampler. ``reset(None)`` continues the sampler;
    an explicit reset seed restarts it for reproducibility. No global NumPy or
    PyTorch random stream is touched by this sampler.
    """

    def __init__(self, config_path, half_angle_degrees, sampling_seed, fixed_angle_degrees=None):
        super().__init__(config_path)
        if self.config.get("attacker", {}).get("policy", {}).get("mode") != "direct":
            raise ValueError("The direction curriculum requires the unchanged Direct Attacker")
        self.half_angle_degrees = _finite_number(half_angle_degrees, "half_angle_degrees")
        if self.half_angle_degrees < 0:
            raise ValueError("half_angle_degrees must be nonnegative")
        self.sampling_seed = _integer(sampling_seed, "sampling_seed", 0)
        self.angle_rng = np.random.default_rng(self.sampling_seed)
        self.fixed_angle_degrees = fixed_angle_degrees
        self.current_angle_degrees = None
        self._reference_initial_states = self._initial_states.copy()
        # Validate the fixed reference now; mutable fixed angles are also
        # revalidated by reset(), as used by a fixed-grid evaluation runner.
        if fixed_angle_degrees is not None:
            self._fixed_angle()

    def _fixed_angle(self):
        value = _finite_number(self.fixed_angle_degrees, "fixed_angle_degrees")
        if abs(value) > self.half_angle_degrees:
            raise ValueError("Fixed evaluation angle exceeds this stage's range")
        return value

    def reset(self, seed=None):
        if seed is not None:
            self.angle_rng = np.random.default_rng(_integer(seed, "seed", 0))
        if self.fixed_angle_degrees is None and self.half_angle_degrees == 0:
            angle = 0.0
        elif self.fixed_angle_degrees is None:
            angle = float(self.angle_rng.uniform(-self.half_angle_degrees, self.half_angle_degrees))
        else:
            angle = self._fixed_angle()
        position = arrival_position(self.config, angle)
        self._initial_states = self._reference_initial_states.copy()
        self._initial_states[-1, :3] = position
        self.current_angle_degrees = angle
        observations, info = super().reset(seed=seed)
        info["arrival_azimuth_offset_degrees"] = angle
        info["initial_attacker_position"] = position.copy()
        return observations, info


def evaluation_angles(half_angle_degrees, episodes):
    """One fixed scene at zero; endpoint-inclusive uniform grids otherwise."""
    half_angle = _finite_number(half_angle_degrees, "half_angle_degrees")
    if half_angle < 0:
        raise ValueError("Evaluation half-angle must be nonnegative")
    if half_angle == 0:
        _integer(episodes, "episodes", 1)
        if episodes != 1:
            raise ValueError("The zero-angle stage evaluates exactly one fixed scene")
        return np.zeros(1, dtype=np.float64)
    count = _integer(episodes, "episodes", 2)
    return np.linspace(-half_angle, half_angle, count, endpoint=True, dtype=np.float64)


def stage_evaluation_settings(curriculum, half_angle_degrees):
    """Return the configured scene count and promotion threshold for a stage."""
    if _finite_number(half_angle_degrees, "half_angle_degrees") == 0:
        if "zero_angle_evaluation_episodes" not in curriculum or "zero_angle_promotion_capture_rate" not in curriculum:
            raise ValueError("Zero-angle stages require explicit evaluation count and success threshold")
        count = _integer(curriculum["zero_angle_evaluation_episodes"], "zero_angle_evaluation_episodes", 1)
        threshold = _rate(curriculum["zero_angle_promotion_capture_rate"], "zero_angle_promotion_capture_rate")
        if count != 1 or threshold != 1.0:
            raise ValueError("Zero-angle promotion requires capture in one deterministic fixed scene")
        return count, threshold
    return (_integer(curriculum["evaluation_episodes"], "evaluation_episodes", 2),
            _rate(curriculum["promotion_capture_rate"], "promotion_capture_rate"))


def _rate(value, name):
    value = _finite_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(name + " must lie in [0,1]")
    return value


def evaluation_key(summary):
    """Rank current-stage evaluations; larger tuples are better.

    Reward and forgetting-monitor results deliberately do not enter the key.
    Unsuccessful capture time remains ``None`` in metrics. Its ranking sentinel
    is zero, used only when both tied capture rates are zero; no Inf is saved.
    """
    try:
        capture = _rate(summary["capture_rate"], "capture_rate")
        enemy_success = _rate(summary["enemy_attack_success_rate"], "enemy_attack_success_rate")
        collision = _rate(summary["collision_episode_rate"], "collision_episode_rate")
        distance = _finite_number(summary["average_min_distance"], "average_min_distance")
        capture_time = summary["average_capture_time"]
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete deterministic evaluation summary") from error
    if distance < 0:
        raise ValueError("average_min_distance must be nonnegative")
    if capture == 0:
        if capture_time is not None:
            raise ValueError("No-capture evaluations must retain null average_capture_time")
        ranked_time = 0.0
    else:
        ranked_time = _finite_number(capture_time, "average_capture_time")
        if ranked_time < 0:
            raise ValueError("average_capture_time must be nonnegative")
    return (capture, -enemy_success, -ranked_time, -distance, -collision)


class PromotionGate:
    """Count consecutive current-stage evaluations meeting the threshold."""

    def __init__(self, threshold, required_consecutive):
        self.threshold = _rate(threshold, "threshold")
        self.required_consecutive = _integer(required_consecutive, "required_consecutive", 1)
        self.consecutive = 0

    def observe(self, capture_rate):
        capture_rate = _rate(capture_rate, "capture_rate")
        self.consecutive = self.consecutive + 1 if capture_rate >= self.threshold else 0
        return self.consecutive >= self.required_consecutive


def _finite_state(value, name):
    if isinstance(value, torch.Tensor):
        valid = bool(torch.isfinite(value).all())
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        valid = bool(np.isfinite(value).all())
    elif isinstance(value, (float, np.floating, complex, np.complexfloating)):
        valid = bool(np.isfinite(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_state(item, name + "." + str(key))
        return
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_state(item, "{}[{}]".format(name, index))
        return
    else:
        return
    if not valid:
        raise ValueError(name + " contains NaN or Inf")


def restore_rng_state(state):
    """Restore global RNGs after loading, including constructor consumption."""
    required = ("python", "numpy", "torch")
    if not isinstance(state, dict) or any(name not in state for name in required):
        raise ValueError("Checkpoint is missing its saved Python/NumPy/Torch RNG state")
    if not isinstance(state["torch"], torch.Tensor):
        raise ValueError("Checkpoint Torch RNG state must be a tensor")
    previous = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        previous["cuda"] = torch.cuda.get_rng_state_all()
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    except (TypeError, ValueError, RuntimeError) as error:
        random.setstate(previous["python"])
        np.random.set_state(previous["numpy"])
        torch.set_rng_state(previous["torch"])
        if "cuda" in previous:
            torch.cuda.set_rng_state_all(previous["cuda"])
        raise ValueError("Invalid checkpoint RNG state") from error


def load_best_eval_checkpoint(path, expected_config, device="cpu"):
    """Load the selected actor, critic and both complete Adam states.

    Stage environments and buffers are intentionally not resumed here. The
    runner starts a new stage with a new range and fresh episode while retaining
    exactly the chosen model/optimizer lineage and its global random state.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported MAPPO checkpoint format")
    for section in ("network", "ppo", "normalization"):
        if section not in expected_config or payload.get("config", {}).get(section) != expected_config[section]:
            raise ValueError("Checkpoint {} settings differ from this curriculum".format(section))
    for name in ("actor", "critic", "actor_optimizer", "critic_optimizer"):
        if not isinstance(payload.get(name), dict):
            raise ValueError("Checkpoint is missing " + name)
        _finite_state(payload[name], name)
    for name in ("actor_optimizer", "critic_optimizer"):
        if "state" not in payload[name] or "param_groups" not in payload[name]:
            raise ValueError("Checkpoint is missing complete Adam state: " + name)
    if "rng_state" not in payload:
        raise ValueError("Checkpoint has no RNG state")
    agent, payload = MAPPOAgent.from_checkpoint(path, device=device)
    restore_rng_state(payload["rng_state"])
    agent.actor.train()
    agent.critic.train()
    agent.actor.requires_grad_(True)
    agent.critic.requires_grad_(True)
    return agent, payload
