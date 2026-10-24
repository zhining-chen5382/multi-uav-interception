"""Configured evader boundary forces and their composition with pursuer forces."""

from copy import deepcopy

import numpy as np
import pytest

from envs.dynamics import MotionLimits
from envs.evader_policy import RepulsiveEvaderPolicy


@pytest.fixture
def boundary_parameters(legacy_config):
    settings = legacy_config["evader"]["policy"]
    bounds = legacy_config["environment"]["bounds"]
    return {
        "bounds_low": np.asarray(bounds["low"], dtype=float),
        "bounds_high": np.asarray(bounds["high"], dtype=float),
        "boundary_margin_horizontal": settings["boundary_margin_horizontal"],
        "boundary_margin_vertical": settings["boundary_margin_vertical"],
        "boundary_strength": settings["boundary_strength"],
    }


@pytest.fixture
def boundary_policy(legacy_config, boundary_parameters):
    return RepulsiveEvaderPolicy(
        MotionLimits.from_config(legacy_config["evader"]),
        legacy_config["evader"]["policy"]["distance_epsilon"],
        **boundary_parameters,
    )


def _center(parameters):
    return (parameters["bounds_low"] + parameters["bounds_high"]) / 2.0


def _margins(parameters):
    return np.array([
        parameters["boundary_margin_horizontal"],
        parameters["boundary_margin_horizontal"],
        parameters["boundary_margin_vertical"],
    ])


def test_boundary_force_is_zero_in_workspace_center(boundary_policy, boundary_parameters):
    assert boundary_policy.boundary_enabled
    force = boundary_policy.boundary_repulsion(_center(boundary_parameters))
    assert force.shape == (3,)
    np.testing.assert_array_equal(force, np.zeros(3))


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
@pytest.mark.parametrize("side", ["low", "high"])
def test_each_boundary_face_has_linear_inward_force(
    boundary_policy, boundary_parameters, axis, side
):
    position = _center(boundary_parameters)
    margin = _margins(boundary_parameters)[axis]
    inward_sign = 1.0 if side == "low" else -1.0
    position[axis] = boundary_parameters["bounds_" + side][axis] + inward_sign * margin / 2.0
    expected = np.zeros(3)
    expected[axis] = inward_sign * boundary_parameters["boundary_strength"] / 2.0
    np.testing.assert_allclose(boundary_policy.boundary_repulsion(position), expected)


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
@pytest.mark.parametrize("side", ["low", "high"])
def test_force_is_continuous_at_soft_margin_entry(
    boundary_policy, boundary_parameters, axis, side
):
    position = _center(boundary_parameters)
    margin = _margins(boundary_parameters)[axis]
    inward_sign = 1.0 if side == "low" else -1.0
    position[axis] = boundary_parameters["bounds_" + side][axis] + inward_sign * margin
    np.testing.assert_array_equal(boundary_policy.boundary_repulsion(position), np.zeros(3))
    delta = margin * 1e-6
    inside = position.copy()
    inside[axis] += inward_sign * delta
    np.testing.assert_array_equal(boundary_policy.boundary_repulsion(inside), np.zeros(3))
    outside = position.copy()
    outside[axis] -= inward_sign * delta
    expected = np.zeros(3)
    expected[axis] = inward_sign * boundary_parameters["boundary_strength"] * 1e-6
    np.testing.assert_allclose(
        boundary_policy.boundary_repulsion(outside), expected, rtol=1e-8, atol=1e-12
    )


@pytest.mark.parametrize("corner", [(0, 0, 0), (1, 1, 1), (0, 1, 0), (1, 0, 1)])
def test_exact_boundary_corners_have_finite_force_and_acceleration(
    boundary_policy, boundary_parameters, corner
):
    low, high = boundary_parameters["bounds_low"], boundary_parameters["bounds_high"]
    position = np.where(corner, high, low)
    expected = np.where(corner, -1.0, 1.0) * boundary_parameters["boundary_strength"]
    force = boundary_policy.boundary_repulsion(position)
    np.testing.assert_allclose(force, expected)
    assert np.isfinite(force).all()
    # Coincident pursuers yield zero pursuer force and isolate the boundary term.
    evader = np.concatenate((position, np.zeros(3)))
    acceleration = boundary_policy(evader, np.tile(evader, (3, 1)))
    assert acceleration.shape == (3,)
    assert np.isfinite(acceleration).all()
    assert np.linalg.norm(acceleration[:2]) <= boundary_policy.limits.max_horizontal_acceleration
    assert abs(acceleration[2]) <= boundary_policy.limits.max_vertical_acceleration


@pytest.mark.parametrize("distance,expected_sign", [(1.0, 1.0), (10.0, -1.0)])
def test_raw_pursuer_force_can_dominate_or_yield_to_boundary_force(
    boundary_policy, boundary_parameters, distance, expected_sign
):
    position = _center(boundary_parameters)
    position[0] = (boundary_parameters["bounds_high"][0]
                   - boundary_parameters["boundary_margin_horizontal"] / 2.0)
    evader = np.concatenate((position, np.zeros(3)))
    pursuers = np.tile(evader, (3, 1))
    pursuers[:, 0] -= distance
    command = boundary_policy(evader, pursuers)
    # Normalizing the pursuer sum before adding the boundary force would give
    # cancellation in both cases, losing the inverse-square threat magnitude.
    np.testing.assert_allclose(
        command, [expected_sign * boundary_policy.limits.max_horizontal_acceleration, 0, 0]
    )


def test_composed_direction_matches_sum_before_normalization(boundary_policy, boundary_parameters):
    position = _center(boundary_parameters)
    position[0] = (boundary_parameters["bounds_high"][0]
                   - boundary_parameters["boundary_margin_horizontal"] / 2.0)
    evader = np.concatenate((position, np.zeros(3)))
    away = np.array([[1.0, 2.0, 3.0], [-4.0, 1.0, 2.0], [2.0, -3.0, 1.0]])
    pursuers = np.zeros((3, 6))
    pursuers[:, :3] = position - away
    pursuer_force = np.sum(away / np.linalg.norm(away, axis=1)[:, None] ** 3, axis=0)
    total_force = pursuer_force + boundary_policy.boundary_repulsion(position)
    limits = boundary_policy.limits
    expected = total_force / np.linalg.norm(total_force) * [
        limits.max_horizontal_acceleration,
        limits.max_horizontal_acceleration,
        limits.max_vertical_acceleration,
    ]
    np.testing.assert_allclose(boundary_policy(evader, pursuers), expected)


def test_legacy_constructor_and_zero_strength_preserve_pursuer_repulsion(
    legacy_config, boundary_parameters
):
    limits = MotionLimits.from_config(legacy_config["evader"])
    epsilon = legacy_config["evader"]["policy"]["distance_epsilon"]
    legacy = RepulsiveEvaderPolicy(limits, epsilon)
    parameters = dict(boundary_parameters, boundary_strength=0.0)
    disabled_force = RepulsiveEvaderPolicy(limits, epsilon, **parameters)
    assert not legacy.boundary_enabled
    np.testing.assert_array_equal(
        legacy.boundary_repulsion(boundary_parameters["bounds_high"]), np.zeros(3)
    )
    evader = np.concatenate((boundary_parameters["bounds_high"], np.zeros(3)))
    pursuers = np.tile(evader, (3, 1))
    pursuers[:, :3] -= [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(disabled_force(evader, pursuers), legacy(evader, pursuers))


@pytest.mark.parametrize("omitted", [
    "bounds_low", "bounds_high", "boundary_margin_horizontal", "boundary_margin_vertical",
    "boundary_strength",
])
def test_partial_boundary_configuration_is_rejected(legacy_config, boundary_parameters, omitted):
    parameters = deepcopy(boundary_parameters)
    del parameters[omitted]
    with pytest.raises(ValueError):
        RepulsiveEvaderPolicy(MotionLimits.from_config(legacy_config["evader"]), 1e-6, **parameters)


@pytest.mark.parametrize("name,value", [
    ("bounds_low", [0.0, 0.0]),
    ("bounds_high", [np.inf, 300.0, 150.0]),
    ("bounds_low", [300.0, -300.0, 0.0]),
    ("boundary_margin_horizontal", 0.0),
    ("boundary_margin_horizontal", 301.0),
    ("boundary_margin_vertical", -1.0),
    ("boundary_margin_vertical", 76.0),
    ("boundary_margin_vertical", np.nan),
    ("boundary_strength", np.inf),
    ("boundary_strength", -0.1),
])
def test_invalid_boundary_values_are_rejected(
    legacy_config, boundary_parameters, name, value
):
    parameters = deepcopy(boundary_parameters)
    parameters[name] = value
    with pytest.raises(ValueError):
        RepulsiveEvaderPolicy(MotionLimits.from_config(legacy_config["evader"]), 1e-6, **parameters)


@pytest.mark.parametrize("position", [
    [0.0, 0.0], [0.0, 0.0, np.nan], [0.0, np.inf, 0.0], [1.0j, 0.0, 0.0],
])
def test_invalid_boundary_positions_are_rejected(boundary_policy, position):
    with pytest.raises(ValueError):
        boundary_policy.boundary_repulsion(position)
