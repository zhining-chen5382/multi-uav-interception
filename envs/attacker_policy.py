"""Direct velocity guidance and the retained historical attacker policy."""

import numpy as np

from .dynamics import MotionLimits, scale_action
from .evader_policy import RepulsiveEvaderPolicy


class DirectAttackerPolicy:
    """Track a constant-speed velocity towards the base, ignoring pursuers.

    Return raw physical acceleration, K * (desired_velocity - velocity).
    Acceleration, jerk and speed constraints belong exclusively to dynamics.
    There is no distance-dependent braking, position hold or boundary force.
    """

    boundary_enabled = False

    def __init__(self, *, base_position, attack_speed, velocity_gain, distance_epsilon):
        self.base_position = RepulsiveEvaderPolicy._finite_vector(
            base_position, "base_position"
        ).copy()
        self.attack_speed = self._positive(attack_speed, "attack_speed")
        self.velocity_gain = self._positive(velocity_gain, "velocity_gain")
        self.distance_epsilon = self._positive(distance_epsilon, "distance_epsilon")

    @staticmethod
    def _positive(value, name):
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("{} must be finite and positive".format(name)) from error
        if not np.isfinite(result) or result <= 0:
            raise ValueError("{} must be finite and positive".format(name))
        return result

    @staticmethod
    def _state(value):
        if np.iscomplexobj(value):
            raise ValueError("attacker_state must contain real, finite values")
        try:
            state = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("attacker_state must contain real, finite values") from error
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("attacker_state must be finite with shape (6,)")
        return state

    def desired_velocity(self, attacker_state):
        """Return attack_speed towards the base; epsilon is a zero-distance guard."""
        state = self._state(attacker_state)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                goal = self.base_position - state[:3]
                distance = np.linalg.norm(goal)
                if distance <= self.distance_epsilon:
                    return np.zeros(3, dtype=np.float64)
                return self.attack_speed * (goal / distance)
            except FloatingPointError as error:
                raise ValueError("Desired velocity exceeds the finite numerical range") from error

    def __call__(self, attacker_state, pursuer_states=None):
        """Return raw acceleration; pursuer_states is intentionally unused."""
        state = self._state(attacker_state)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                return self.velocity_gain * (self.desired_velocity(state) - state[3:])
            except FloatingPointError as error:
                raise ValueError("Velocity tracking command exceeds the finite numerical range") from error


class GoalDirectedAttackerPolicy(RepulsiveEvaderPolicy):
    """Historical goal-attraction policy, retained for old environment snapshots.

    The goal term has unit magnitude outside distance_epsilon. The avoidance
    term retains the original, unnormalized inverse-square pursuer repulsion:
    a single pursuer contributes magnitude 1 / distance**2. Only the combined
    direction is normalized and mapped to the existing physical acceleration
    limits. Exact cancellation returns zero, without random tie-breaking.

    The parent supplies only boundary configuration/validation and its optional
    boundary force. Its evasion-only action method is fully replaced here.
    Omitting all five boundary arguments disables soft boundary steering.
    """

    def __init__(
        self,
        limits: MotionLimits,
        distance_epsilon: float,
        *,
        base_position: np.ndarray,
        goal_weight: float,
        avoidance_weight: float,
        bounds_low: np.ndarray = None,
        bounds_high: np.ndarray = None,
        boundary_margin_horizontal: float = None,
        boundary_margin_vertical: float = None,
        boundary_strength: float = None,
    ) -> None:
        super().__init__(
            limits,
            distance_epsilon,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            boundary_margin_horizontal=boundary_margin_horizontal,
            boundary_margin_vertical=boundary_margin_vertical,
            boundary_strength=boundary_strength,
        )
        self.base_position = self._finite_vector(base_position, "base_position").copy()
        self.goal_weight = self._weight(goal_weight, "goal_weight", positive=True)
        self.avoidance_weight = self._weight(
            avoidance_weight, "avoidance_weight", positive=False
        )

    @staticmethod
    def _weight(value: float, name: str, *, positive: bool) -> float:
        requirement = "positive" if positive else "nonnegative"
        try:
            weight = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("{} must be finite and {}".format(name, requirement)) from error
        if not np.isfinite(weight) or (weight <= 0 if positive else weight < 0):
            raise ValueError("{} must be finite and {}".format(name, requirement))
        return weight

    def __call__(self, attacker_state: np.ndarray, pursuer_states: np.ndarray) -> np.ndarray:
        """Return a finite physical acceleration command with shape (3,)."""
        if np.iscomplexobj(attacker_state) or np.iscomplexobj(pursuer_states):
            raise ValueError("Aircraft states must contain real, finite values")
        try:
            attacker_state = np.asarray(attacker_state, dtype=np.float64)
            pursuer_states = np.asarray(pursuer_states, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Aircraft states must contain real, finite values") from error
        if attacker_state.shape != (6,) or pursuer_states.shape != (3, 6):
            raise ValueError("attacker_state and pursuer_states must have shapes (6,) and (3, 6)")
        if not np.all(np.isfinite(attacker_state)) or not np.all(np.isfinite(pursuer_states)):
            raise ValueError("Aircraft states must contain real, finite values")

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                goal = self.base_position - attacker_state[:3]
                goal_distance = np.linalg.norm(goal)
                goal_direction = (
                    goal / goal_distance
                    if goal_distance > self.distance_epsilon
                    else np.zeros(3, dtype=np.float64)
                )
                away = attacker_state[:3] - pursuer_states[:, :3]
                distance = np.linalg.norm(away, axis=1)
                denominator = np.maximum(distance, self.distance_epsilon) ** 3
                pursuer_repulsion = np.sum(away / denominator[:, np.newaxis], axis=0)
                combined = (
                    self.goal_weight * goal_direction
                    + self.avoidance_weight * pursuer_repulsion
                    + self.boundary_repulsion(attacker_state[:3])
                )
            except FloatingPointError as error:
                raise ValueError("Attacker steering exceeds the finite numerical range") from error

        # Rescale before computing the norm so a finite, strong near-contact
        # force cannot overflow during normalization.
        maximum = np.max(np.abs(combined))
        if maximum == 0:
            return np.zeros(3, dtype=np.float64)
        direction = combined / maximum
        direction = direction / np.linalg.norm(direction)
        return scale_action(direction, self.limits)
