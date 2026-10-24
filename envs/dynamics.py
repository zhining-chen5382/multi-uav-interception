"""Configurable 3D point-mass dynamics for Baseline v1."""

from dataclasses import dataclass, fields
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class MotionLimits:
    """Physical limits in metres, seconds, and their derived units."""

    max_horizontal_speed: float
    max_climb_speed: float
    max_descent_speed: float
    max_horizontal_acceleration: float
    max_vertical_acceleration: float
    max_jerk: float

    def __post_init__(self) -> None:
        for field in fields(self):
            try:
                value = float(getattr(self, field.name))
            except (TypeError, ValueError) as error:
                raise ValueError("{} must be finite and positive".format(field.name)) from error
            if not np.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and positive".format(field.name))
            object.__setattr__(self, field.name, value)

    @classmethod
    def from_config(cls, config: Mapping[str, float]) -> "MotionLimits":
        """Read the six required limit values from a YAML configuration section."""
        try:
            return cls(**{field.name: config[field.name] for field in fields(cls)})
        except KeyError as error:
            raise ValueError("Missing motion limit: {}".format(error.args[0])) from error


def _finite_array(value: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("{} must contain real, finite values".format(name))
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must contain real, finite values".format(name)) from error
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain real, finite values".format(name))
    return array


def _limit_horizontal(vector: np.ndarray, maximum: float) -> np.ndarray:
    """Return a copy with the horizontal vector's magnitude limited."""
    result = vector.copy()
    horizontal_norm = np.hypot(result[..., 0], result[..., 1])
    scale = np.ones_like(horizontal_norm)
    np.divide(maximum, horizontal_norm, out=scale, where=horizontal_norm > maximum)
    result[..., :2] *= scale[..., np.newaxis]
    return result


def _limit_acceleration(acceleration: np.ndarray, limits: MotionLimits) -> np.ndarray:
    result = _limit_horizontal(acceleration, limits.max_horizontal_acceleration)
    result[..., 2] = np.clip(
        result[..., 2], -limits.max_vertical_acceleration, limits.max_vertical_acceleration
    )
    return result


def scale_action(actions: np.ndarray, limits: MotionLimits) -> np.ndarray:
    """Map finite normalized actions of shape (..., 3) to physical acceleration.

    Each component is clipped to [-1, 1] before scaling. The horizontal
    acceleration then receives a vector-magnitude constraint.
    """
    actions = _finite_array(actions, "actions")
    if actions.ndim < 1 or actions.shape[-1] != 3:
        raise ValueError("actions must have shape (..., 3)")
    acceleration = np.clip(actions, -1.0, 1.0) * np.array(
        [
            limits.max_horizontal_acceleration,
            limits.max_horizontal_acceleration,
            limits.max_vertical_acceleration,
        ],
        dtype=np.float64,
    )
    return _limit_acceleration(acceleration, limits)


def integrate(
    state: np.ndarray,
    acceleration_command: np.ndarray,
    previous_acceleration: np.ndarray,
    dt: float,
    limits: MotionLimits,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance state [..., x, y, z, vx, vy, vz] by one fixed time step.

    Command limits precede a 3D Euclidean jerk limit. Velocity limits apply
    before the semi-implicit position update. There is no boundary handling.

    Pass zero as previous_acceleration at initialization and then reuse the
    returned applied_acceleration. It is the acceleration command after the
    acceleration and jerk constraints; speed clipping can make actual dv/dt
    different from this command.
    """
    state = _finite_array(state, "state")
    acceleration_command = _finite_array(acceleration_command, "acceleration_command")
    previous_acceleration = _finite_array(previous_acceleration, "previous_acceleration")
    if state.ndim < 1 or state.shape[-1] != 6:
        raise ValueError("state must have shape (..., 6)")
    expected_shape = state.shape[:-1] + (3,)
    if acceleration_command.shape != expected_shape:
        raise ValueError("acceleration_command must have shape {}".format(expected_shape))
    if previous_acceleration.shape != expected_shape:
        raise ValueError("previous_acceleration must have shape {}".format(expected_shape))
    try:
        dt = float(dt)
    except (TypeError, ValueError) as error:
        raise ValueError("dt must be finite and positive") from error
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    acceleration = _limit_acceleration(acceleration_command, limits)
    change = acceleration - previous_acceleration
    change_norm = np.linalg.norm(change, axis=-1, keepdims=True)
    maximum_change = limits.max_jerk * dt
    change_scale = np.ones_like(change_norm)
    np.divide(maximum_change, change_norm, out=change_scale, where=change_norm > maximum_change)
    applied_acceleration = previous_acceleration + change * change_scale

    velocity = state[..., 3:] + applied_acceleration * dt
    velocity = _limit_horizontal(velocity, limits.max_horizontal_speed)
    velocity[..., 2] = np.clip(
        velocity[..., 2], -limits.max_descent_speed, limits.max_climb_speed
    )
    position = state[..., :3] + velocity * dt
    next_state = np.concatenate((position, velocity), axis=-1)
    return next_state, applied_acceleration
