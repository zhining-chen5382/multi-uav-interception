"""Deterministic rule-based evasion for the three-pursuer Baseline v1."""

import numpy as np

from .dynamics import MotionLimits


class RepulsiveEvaderPolicy:
    """Combine inverse-square pursuer repulsion with soft boundary steering.

    The configurable distance_epsilon regularizes coincident or very close
    positions. Boundary forces apply only to the evader, before normalizing
    the total direction; environment hard clipping remains a numerical safety.
    Exact cancellation yields a zero command, without random tie-breaking.
    Omitting every boundary parameter preserves the legacy pursuer-only policy.
    """

    def __init__(
        self,
        limits: MotionLimits,
        distance_epsilon: float,
        *,
        bounds_low: np.ndarray = None,
        bounds_high: np.ndarray = None,
        boundary_margin_horizontal: float = None,
        boundary_margin_vertical: float = None,
        boundary_strength: float = None,
    ) -> None:
        self.limits = limits
        try:
            self.distance_epsilon = float(distance_epsilon)
        except (TypeError, ValueError) as error:
            raise ValueError("distance_epsilon must be finite and positive") from error
        if not np.isfinite(self.distance_epsilon) or self.distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be finite and positive")

        boundary_parameters = (
            bounds_low,
            bounds_high,
            boundary_margin_horizontal,
            boundary_margin_vertical,
            boundary_strength,
        )
        self.boundary_enabled = all(value is not None for value in boundary_parameters)
        if not self.boundary_enabled:
            if any(value is not None for value in boundary_parameters):
                raise ValueError("All five boundary parameters must be provided together")
            return

        self.bounds_low = self._finite_vector(bounds_low, "bounds_low").copy()
        self.bounds_high = self._finite_vector(bounds_high, "bounds_high").copy()
        if np.any(self.bounds_low >= self.bounds_high):
            raise ValueError("bounds_low must be strictly below bounds_high on every axis")
        try:
            horizontal = float(boundary_margin_horizontal)
            vertical = float(boundary_margin_vertical)
            self.boundary_strength = float(boundary_strength)
        except (TypeError, ValueError) as error:
            raise ValueError("Boundary margins and strength must be real, finite numbers") from error
        self.boundary_margins = np.array([horizontal, horizontal, vertical], dtype=np.float64)
        if not np.all(np.isfinite(self.boundary_margins)) or np.any(self.boundary_margins <= 0):
            raise ValueError("Boundary margins must be finite and positive")
        if not np.isfinite(self.boundary_strength) or self.boundary_strength < 0:
            raise ValueError("boundary_strength must be finite and nonnegative")
        half_span = self.bounds_high / 2.0 - self.bounds_low / 2.0
        if np.any(self.boundary_margins > half_span):
            raise ValueError("Boundary margins cannot exceed half of their axis span")

    @staticmethod
    def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
        if np.iscomplexobj(value):
            raise ValueError("{} must contain real, finite values".format(name))
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("{} must contain real, finite values".format(name)) from error
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("{} must have shape (3,) and contain real, finite values".format(name))
        return vector

    def boundary_repulsion(self, position: np.ndarray) -> np.ndarray:
        """Return continuous inward steering within each configured soft margin.

        The horizontal margin applies to x/y and the vertical margin to z.
        No extra force clipping or velocity-dependent steering is applied.
        """
        position = self._finite_vector(position, "position")
        if not self.boundary_enabled or self.boundary_strength == 0:
            return np.zeros(3, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                lower = np.maximum(self.bounds_low + self.boundary_margins - position, 0.0)
                upper = np.maximum(position - (self.bounds_high - self.boundary_margins), 0.0)
                repulsion = self.boundary_strength * ((lower - upper) / self.boundary_margins)
            except FloatingPointError as error:
                raise ValueError("Boundary repulsion exceeds the finite numerical range") from error
        return repulsion

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
        repulsion = repulsion + self.boundary_repulsion(evader_state[:3])
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
