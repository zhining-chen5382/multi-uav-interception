"""Linear coefficient sweeps for previously verified offline reward components."""

from collections.abc import Mapping

import numpy as np


IDENTITY_FIELDS = (
    "dataset", "cohort", "episode_id", "duplicate_of", "training_seed", "model",
    "checkpoint_step", "scenario_id",
)
COMPONENT_FIELDS = (
    "original_return", "original_return_discounted", "active_near_return",
    "active_near_return_discounted", "line_return", "line_return_discounted",
)
REQUIRED_FIELDS = IDENTITY_FIELDS + COMPONENT_FIELDS + (
    "angle_degrees", "min_distance", "captured", "defense_failure", "timeout",
    "event_time", "category",
)


def _number(value, name):
    """Accept scalar CSV numbers while rejecting arrays, booleans and NaN/Inf."""
    try:
        scalar = np.asarray(value)
        if scalar.ndim != 0 or isinstance(value, (bool, np.bool_)):
            raise ValueError
        result = float(scalar)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + " must be a finite numeric scalar") from None
    if not np.isfinite(result):
        raise ValueError(name + " must be a finite numeric scalar")
    return result


def _boolean(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in ("True", "False"):
        return value == "True"
    raise ValueError(name + " must be a boolean or the CSV string True/False")


def score_rows(rows, beta, k, source_beta, source_k):
    """Rescale corrected near and line components without changing trajectories.

    Source rows contain the active-near replay at ``source_beta`` and the line
    cost at ``source_k``. Discounted sums rescale by the same coefficients,
    because the recorded timing and discount factor stay fixed.
    """
    beta = _number(beta, "beta")
    k = _number(k, "k")
    source_beta = _number(source_beta, "source_beta")
    source_k = _number(source_k, "source_k")
    if beta < 0 or k < 0 or source_beta <= 0 or source_k <= 0:
        raise ValueError("beta and k must be nonnegative; source coefficients must be positive")
    near_scale = beta / source_beta
    line_scale = k / source_k
    if not np.isfinite([near_scale, line_scale]).all():
        raise ValueError("Coefficient ratios exceed the supported numerical range")
    try:
        iterator = iter(rows)
    except TypeError:
        raise ValueError("rows must be an iterable of component mappings") from None

    results = []
    for row in iterator:
        if not isinstance(row, Mapping) or any(field not in row for field in REQUIRED_FIELDS):
            raise ValueError("Each source row must contain all component and identity fields")
        values = {field: _number(row[field], field) for field in COMPONENT_FIELDS}
        if (values["active_near_return"] < 0 or values["active_near_return_discounted"] < 0
                or values["line_return"] > 0 or values["line_return_discounted"] > 0):
            raise ValueError("Source near components must be nonnegative and line components nonpositive")

        result = dict(beta=beta, k=k)
        result.update({field: row[field] for field in IDENTITY_FIELDS})
        result.update(
            angle_degrees=_number(row["angle_degrees"], "angle_degrees"),
            min_distance=_number(row["min_distance"], "min_distance"),
            captured=_boolean(row["captured"], "captured"),
            defense_failure=_boolean(row["defense_failure"], "defense_failure"),
            timeout=_boolean(row["timeout"], "timeout"),
            event_time=_number(row["event_time"], "event_time"),
            original_return=values["original_return"],
            original_discounted_return=values["original_return_discounted"],
            near_return=near_scale * values["active_near_return"],
            near_discounted_return=near_scale * values["active_near_return_discounted"],
            line_return=line_scale * values["line_return"],
            line_discounted_return=line_scale * values["line_return_discounted"],
        )
        if result["min_distance"] < 0 or result["event_time"] < 0:
            raise ValueError("Recorded distance and event time must be nonnegative")
        result["total_return"] = result["original_return"] + result["near_return"] + result["line_return"]
        result["discounted_return"] = (result["original_discounted_return"]
                                       + result["near_discounted_return"]
                                       + result["line_discounted_return"])
        numeric = ("near_return", "near_discounted_return", "line_return",
                   "line_discounted_return", "total_return", "discounted_return")
        if not np.isfinite([result[field] for field in numeric]).all():
            raise ValueError("Rescaled reward exceeds the supported numerical range")
        result["category"] = row["category"]
        results.append(result)
    return results
