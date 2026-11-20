"""Configurable reward arithmetic, independent of environment dynamics."""

import math


# The other returned fields are distances for diagnostics, not reward terms.
ACTIVE_REWARD_TERMS = (
    "active_progress_reward", "survival", "collision", "capture", "attack_failure",
)
ABSOLUTE_DISTANCE_REWARD_TERMS = (
    "absolute_distance_cost", "survival", "collision", "capture", "attack_failure",
)


def _finite_scalar(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite scalar".format(name)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be a finite scalar".format(name))
    return result


def absolute_distance_components(
    min_distance, dt, coefficients, captured=False, attacker_reached_base=False,
    collision_pairs=0,
):
    """Charge the post-update nearest distance for the elapsed physical time.

    ``distance_time_integral`` is a right-endpoint rectangle, in metre-seconds.
    Sum only ``ABSOLUTE_DISTANCE_REWARD_TERMS``: the unweighted distance and
    distance-time diagnostics are not additional reward terms. Capture retains
    priority over an attacker reaching the base on the same transition.
    """
    min_distance = _finite_scalar(min_distance, "min_distance")
    if min_distance < 0:
        raise ValueError("min_distance must be nonnegative")
    dt = _finite_scalar(dt, "dt")
    if dt <= 0:
        raise ValueError("dt must be positive")
    weights = {
        name: _finite_scalar(coefficients[name], "reward." + name)
        for name in (
            "distance_cost_coefficient", "survival_penalty_per_second",
            "collision_penalty", "capture_reward", "attack_failure_penalty",
        )
    }
    if weights["distance_cost_coefficient"] <= 0:
        raise ValueError("reward.distance_cost_coefficient must be positive")
    distance_time_integral = min_distance * dt
    components = {
        "distance_to_nearest_pursuer": min_distance,
        "distance_time_integral": distance_time_integral,
        "absolute_distance_cost": -weights["distance_cost_coefficient"] * distance_time_integral,
        "survival": weights["survival_penalty_per_second"] * dt,
        "collision": weights["collision_penalty"] if collision_pairs else 0.0,
        "capture": weights["capture_reward"] if captured else 0.0,
        "attack_failure": (
            weights["attack_failure_penalty"]
            if attacker_reached_base and not captured else 0.0
        ),
    }
    if not all(math.isfinite(value) for value in components.values()):
        raise ValueError("Reward computation exceeds the finite numerical range")
    return components


def active_progress_components(
    previous_min_distance, min_distance, previous_base_distance, base_distance,
    dt, coefficients, captured=False, attacker_reached_base=False, collision_pairs=0,
):
    """Return raw progress diagnostics and five weighted team reward terms.

    Distances are measured at the same pre/post-step instants. Survival charges
    the elapsed physical time, including the terminal transition. Callers must
    sum only ``ACTIVE_REWARD_TERMS``; raw progress fields are not extra rewards.
    """
    distances = [
        _finite_scalar(value, name) for name, value in (
            ("previous_min_distance", previous_min_distance),
            ("min_distance", min_distance),
            ("previous_base_distance", previous_base_distance),
            ("base_distance", base_distance),
        )
    ]
    if any(value < 0 for value in distances):
        raise ValueError("Reward distances must be nonnegative")
    dt = _finite_scalar(dt, "dt")
    if dt <= 0:
        raise ValueError("dt must be positive")
    weights = {
        name: _finite_scalar(coefficients[name], "reward." + name)
        for name in (
            "active_progress_coefficient", "survival_penalty_per_second",
            "collision_penalty", "capture_reward", "attack_failure_penalty",
        )
    }
    pursuer_progress = distances[0] - distances[1]
    base_progress = distances[2] - distances[3]
    active_progress = pursuer_progress - base_progress
    components = {
        "pursuer_attacker_progress": pursuer_progress,
        "attacker_base_progress": base_progress,
        "active_progress": active_progress,
        "active_progress_reward": weights["active_progress_coefficient"] * active_progress,
        "survival": weights["survival_penalty_per_second"] * dt,
        "collision": weights["collision_penalty"] if collision_pairs else 0.0,
        "capture": weights["capture_reward"] if captured else 0.0,
        "attack_failure": (
            weights["attack_failure_penalty"]
            if attacker_reached_base and not captured else 0.0
        ),
    }
    if not all(math.isfinite(value) for value in components.values()):
        raise ValueError("Reward computation exceeds the finite numerical range")
    return components
