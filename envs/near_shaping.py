"""Shared record-distance shaping attributed to actual pursuer displacement."""

import numpy as np


def motion_attributed_components(previous_best, next_best, counterfactual_distance,
                                 length_scale, beta):
    """Compare the actual new record with a frozen-pursuer one-step baseline.

    The attacker position is the recorded next position in both distances.
    Scalars and broadcast-compatible arrays use the same arithmetic. Actual
    historical minima are retained even when their improvement earns no reward.
    """
    inputs = (previous_best, next_best, counterfactual_distance, length_scale, beta)
    if any(np.iscomplexobj(value) for value in inputs):
        raise ValueError("Near shaping requires real finite distances and scalars")
    try:
        previous_best, next_best, counterfactual_distance = np.broadcast_arrays(
            np.asarray(previous_best, dtype=np.float64), np.asarray(next_best, dtype=np.float64),
            np.asarray(counterfactual_distance, dtype=np.float64))
        length_scale = np.asarray(length_scale, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Near shaping requires compatible finite distances and scalars") from error
    if (length_scale.ndim != 0 or beta.ndim != 0
            or not np.isfinite(length_scale) or length_scale <= 0
            or not np.isfinite(beta) or beta < 0):
        raise ValueError("length_scale must be positive and beta nonnegative finite scalars")
    if (not all(np.isfinite(value).all() for value in (previous_best, next_best, counterfactual_distance))
            or np.any(previous_best < 0) or np.any(next_best < 0)
            or np.any(counterfactual_distance < 0) or np.any(next_best > previous_best)):
        raise ValueError("Near shaping distances must be finite, nonnegative and retain the historical minimum")
    counterfactual_best = np.minimum(previous_best, counterfactual_distance)
    with np.errstate(over="ignore", under="ignore"):
        old_potential = np.exp(-previous_best / length_scale)
        next_potential = np.exp(-next_best / length_scale)
        counterfactual_potential = np.exp(-counterfactual_best / length_scale)
    original = beta * (next_potential - old_potential)
    active = beta * np.maximum(0.0, next_potential - counterfactual_potential)
    return dict(original_near=original, active_near=active, removed_near=original - active,
                counterfactual_best_distances=counterfactual_best)
