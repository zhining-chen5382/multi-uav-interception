"""Deterministic rule-based evasion for the three-pursuer Baseline v1."""

import numpy as np

from .dynamics import MotionLimits


class RepulsiveEvaderPolicy:
    """Accelerate away from the sum of inverse-square pursuer repulsions.

    The configurable distance_epsilon regularizes coincident or very close
    positions. Exact force cancellation yields a zero command; there is no
    random tie-breaking, boundary steering, or trajectory prediction.
    """

    def __init__(self, limits: MotionLimits, distance_epsilon: float) -> None:
        self.limits = limits
        try:
            self.distance_epsilon = float(distance_epsilon)
        except (TypeError, ValueError) as error:
            raise ValueError("distance_epsilon must be finite and positive") from error
        if not np.isfinite(self.distance_epsilon) or self.distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be finite and positive")

    def __call__(self, evader_state: np.ndarray, pursuer_states: np.ndarray) -> np.ndarray:
        """Return a physical acceleration command with shape (3,)."""
        if np.iscomplexobj(evader_state) or np.iscomplexobj(pursuer_states):
            raise ValueError("Aircraft states must contain real, finite values")
        try:
            evader_state = np.asarray(evader_state, dtype=np.float64)
            pursuer_states = np.asarray(pursuer_states, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Aircraft states must contain real, finite values") from error
        if evader_state.shape != (6,) or pursuer_states.shape != (3, 6):
            raise ValueError("evader_state and pursuer_states must have shapes (6,) and (3, 6)")
        if not np.all(np.isfinite(evader_state)) or not np.all(np.isfinite(pursuer_states)):
            raise ValueError("Aircraft states must contain real, finite values")

        away = evader_state[:3] - pursuer_states[:, :3]
        distance = np.linalg.norm(away, axis=1)
        denominator = np.maximum(distance, self.distance_epsilon) ** 3
        repulsion = np.sum(away / denominator[:, np.newaxis], axis=0)
        magnitude = np.linalg.norm(repulsion)
        if magnitude == 0:
            return np.zeros(3, dtype=np.float64)
        direction = repulsion / magnitude
        return direction * np.array(
            [
                self.limits.max_horizontal_acceleration,
                self.limits.max_horizontal_acceleration,
                self.limits.max_vertical_acceleration,
            ],
            dtype=np.float64,
        )
