"""Geometry and reward arithmetic for offline interception-line calibration."""

import numpy as np

from envs.geometry import segment_distances
from scripts.offline_best_so_far_reward_replay import best_so_far_shaping
from scripts.offline_near_reward_replay import reward_components


def line_cost(distances, time, coefficient):
    """Apply -k*d_post*dt to recorded states, including the terminal step."""
    distances = np.asarray(distances, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if (distances.ndim != 1 or len(distances) < 2 or time.shape != distances.shape
            or not np.isfinite(distances).all() or np.any(distances < 0)
            or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)
            or np.ndim(coefficient) != 0 or not np.isfinite(coefficient) or coefficient < 0):
        raise ValueError("Invalid distance series, timestamps or nonnegative coefficient")
    result = -float(coefficient) * distances[1:] * np.diff(time)
    if not np.isfinite(result).all():
        raise ValueError("Line cost exceeds the supported numerical range")
    return result


def episode_components(states, time, environment):
    """Reconstruct the existing reward and verify the first terminal sample."""
    states = np.asarray(states, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if (states.ndim != 3 or states.shape[1:] != (2, 6) or len(states) < 2
            or time.shape != (len(states),) or not np.isfinite(states).all()):
        raise ValueError("A complete finite two-aircraft trajectory is required")
    dt = float(environment["environment"]["dt"])
    np.testing.assert_allclose(time, np.arange(len(states)) * dt, rtol=0, atol=1e-10)
    base = np.asarray(environment["mission"]["base_position"], dtype=np.float64)
    distances = np.linalg.norm(states[:, 0, :3] - states[:, 1, :3], axis=1)
    base_distances = np.linalg.norm(states[:, 1, :3] - base, axis=1)
    captures = distances < environment["capture"]["radius"]
    failures = (~captures) & (base_distances < environment["mission"]["attack_radius"])
    timeouts = np.arange(len(states)) >= environment["environment"]["max_episode_steps"]
    events = np.flatnonzero((captures | failures | timeouts)[1:]) + 1
    if not len(events) or int(events[0]) != len(states) - 1:
        raise ValueError("Trajectory must end at its first terminal sample")
    captured, failed = bool(captures[-1]), bool(failures[-1])
    names = ("distance_cost_coefficient", "survival_penalty_per_second",
             "capture_reward", "attack_failure_penalty")
    coefficients = {name: environment["reward"][name] for name in names}
    components = reward_components(distances, dt, captured, failed, coefficients)
    shaping = environment["reward"]["best_so_far_shaping"]
    components["near_shaping"] = best_so_far_shaping(
        distances, shaping["length_scale_m"], shaping["beta"])
    components["total"] = components["original"] + components["near_shaping"]
    return dict(states=states, time=time, components=components, distances=distances,
                captured=captured, defense_failure=failed,
                timeout=bool(not (captured or failed) and timeouts[-1]),
                event_index=len(states) - 1)


def stationary_counterfactual(states, time, environment):
    """Keep P0 stationary and reuse an independent saved Direct Attacker path."""
    if environment["attacker"]["policy"]["mode"] != "direct":
        raise ValueError("Stationary replay requires an independent Direct Attacker")
    initialization = environment["initialization"]
    if np.any(np.asarray(initialization["pursuer_velocity"], dtype=float) != 0):
        raise ValueError("Stationary replay requires a zero-velocity initial pursuer")
    states = np.asarray(states, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if (states.ndim != 3 or states.shape[1:] != (2, 6) or len(states) < 2
            or time.shape != (len(states),) or not np.isfinite(states).all()):
        raise ValueError("Counterfactual source must be a finite complete state series")
    result = states.copy()
    result[:, 0, :3] = initialization["pursuer_position"]
    result[:, 0, 3:] = 0
    distances = np.linalg.norm(result[:, 0, :3] - result[:, 1, :3], axis=1)
    base = environment["mission"]["base_position"]
    base_distances = np.linalg.norm(result[:, 1, :3] - base, axis=1)
    events = ((distances < environment["capture"]["radius"])
              | (base_distances < environment["mission"]["attack_radius"])
              | (np.arange(len(result)) >= environment["environment"]["max_episode_steps"]))
    indices = np.flatnonzero(events[1:]) + 1
    if not len(indices):
        raise ValueError("Saved attacker path does not cover the counterfactual terminal event")
    end = int(indices[0]) + 1
    return episode_components(result[:end], time[:end].copy(), environment)
