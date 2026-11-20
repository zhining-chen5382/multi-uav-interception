"""Position-attributed record-distance shaping for frozen trajectory replay."""

import numpy as np

from envs.near_shaping import motion_attributed_components


def active_near_shaping(states, length_scale, beta):
    """Remove the reward obtainable by holding P0 at its pre-step position.

    Each transition reuses the recorded next attacker position. The comparison
    therefore attributes the reward to P0's actual displacement, including
    inertial motion. All history minima use the recorded actual trajectory.
    """
    states = np.asarray(states, dtype=np.float64)
    if (states.ndim != 3 or states.shape[1:] != (2, 6) or len(states) < 2
            or not np.isfinite(states).all()):
        raise ValueError("A finite (T+1, 2, 6) trajectory with T >= 1 is required")
    length_scale = np.asarray(length_scale, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if (length_scale.ndim != 0 or beta.ndim != 0
            or not np.isfinite(length_scale) or length_scale <= 0
            or not np.isfinite(beta) or beta < 0):
        raise ValueError("length_scale must be positive and beta nonnegative finite scalars")

    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(states[:, 1, :3] - states[:, 0, :3], axis=1)
        counterfactual_distances = np.linalg.norm(
            states[1:, 1, :3] - states[:-1, 0, :3], axis=1)
    if (not np.isfinite(distances).all()
            or not np.isfinite(counterfactual_distances).all()):
        raise ValueError("Trajectory distances exceed the supported numerical range")

    best_distances = np.minimum.accumulate(distances)
    components = motion_attributed_components(best_distances[:-1], best_distances[1:],
                                              counterfactual_distances, length_scale, beta)
    return dict(
        distances=distances,
        best_distances=best_distances,
        counterfactual_distances=counterfactual_distances,
        **components,
    )
