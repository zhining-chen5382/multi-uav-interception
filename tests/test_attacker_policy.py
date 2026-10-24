"""Goal-directed attack, inverse-square avoidance, and numerical edge cases."""

import numpy as np
import pytest

from envs.attacker_policy import GoalDirectedAttackerPolicy
from envs.dynamics import MotionLimits, integrate, scale_action


@pytest.fixture
def limits(baseline_config):
    return MotionLimits.from_config(baseline_config["pursuer"])


@pytest.fixture
def policy(limits):
    return GoalDirectedAttackerPolicy(
        limits, 1e-6, base_position=np.zeros(3), goal_weight=1.0, avoidance_weight=25.0
    )


def _state(position):
    return np.concatenate((np.asarray(position, dtype=np.float64), np.zeros(3)))


def _pursuers(positions):
    return np.column_stack((np.asarray(positions, dtype=np.float64), np.zeros((3, 3))))


def test_no_avoidance_points_towards_configured_base(limits):
    base = np.array([10.0, -20.0, 30.0])
    policy = GoalDirectedAttackerPolicy(
        limits, 1e-6, base_position=base, goal_weight=2.0, avoidance_weight=0.0
    )
    attacker = _state([110.0, 80.0, 130.0])
    command = policy(attacker, np.zeros((3, 6)))
    expected = scale_action(-np.ones(3) / np.sqrt(3), limits)
    np.testing.assert_allclose(command, expected)
    assert not policy.boundary_enabled


def test_far_pursuers_do_not_cancel_attack_goal(policy):
    attacker = _state([500.0, 0.0, 20.0])
    pursuers = _pursuers([[0, -1, 1], [0, 1, 1], [1, 0, 1]])
    command = policy(attacker, pursuers)
    assert command[0] < 0
    assert np.dot(command, policy.base_position - attacker[:3]) > 0
    np.testing.assert_array_equal(command, policy(attacker, pursuers))


def test_side_threat_deflects_attack_without_abandoning_goal(policy):
    attacker = _state([100, 0, 20])
    pursuers = _pursuers([[100, 10, 20], [100, -1000, 20], [100, 0, 1020]])
    command = policy(attacker, pursuers)
    assert command[0] < 0
    assert command[1] < 0
    assert abs(command[1]) < abs(command[0])
    assert np.dot(command, policy.base_position - attacker[:3]) > 0


def test_combined_direction_preserves_raw_inverse_square_avoidance(policy):
    attacker = _state([100, 20, 40])
    away = np.array([[10, 2, 1], [-3, 20, 2], [2, -4, 30]], dtype=np.float64)
    pursuers = _pursuers(attacker[:3] - away)
    goal = -attacker[:3] / np.linalg.norm(attacker[:3])
    repulsion = np.sum(away / np.linalg.norm(away, axis=1)[:, None] ** 3, axis=0)
    combined = policy.goal_weight * goal + policy.avoidance_weight * repulsion
    expected = scale_action(combined / np.linalg.norm(combined), policy.limits)
    np.testing.assert_allclose(policy(attacker, pursuers), expected)


def test_custom_acceleration_limits_control_the_physical_command(baseline_config):
    settings = dict(baseline_config["pursuer"])
    settings.update(max_horizontal_acceleration=1.2, max_vertical_acceleration=0.7)
    custom_limits = MotionLimits.from_config(settings)
    policy = GoalDirectedAttackerPolicy(
        custom_limits, 1e-6, base_position=[0, 0, 0],
        goal_weight=1.0, avoidance_weight=0.0,
    )
    attacker = _state([100, 100, 100])
    command = policy(attacker, np.zeros((3, 6)))
    np.testing.assert_allclose(command, -np.array([1.2, 1.2, 0.7]) / np.sqrt(3))
    assert np.linalg.norm(command[:2]) <= custom_limits.max_horizontal_acceleration
    assert abs(command[2]) <= custom_limits.max_vertical_acceleration


def test_exact_goal_avoidance_cancellation_is_zero(limits):
    # The single noncoincident pursuer contributes +1 along x; the goal is -x.
    policy = GoalDirectedAttackerPolicy(
        limits, 1e-6, base_position=[0, 0, 0], goal_weight=1.0, avoidance_weight=1.0
    )
    attacker = _state([100, 0, 0])
    pursuers = _pursuers([[99, 0, 0], [100, 0, 0], [100, 0, 0]])
    np.testing.assert_array_equal(policy(attacker, pursuers), np.zeros(3))


@pytest.mark.parametrize("offset", [0.0, 1e-12, 1e-6])
def test_at_or_within_epsilon_of_base_has_safe_zero_goal(limits, offset):
    policy = GoalDirectedAttackerPolicy(
        limits, 1e-6, base_position=[0, 0, 0], goal_weight=1.0, avoidance_weight=0.0
    )
    attacker = _state([offset, 0, 0])
    result = policy(attacker, np.tile(attacker, (3, 1)))
    np.testing.assert_array_equal(result, np.zeros(3))
    assert np.isfinite(result).all()


@pytest.mark.parametrize("separation", [0.0, 1e-12, 1e-7, 1e-6])
def test_coincident_and_very_close_pursuers_remain_finite(policy, separation):
    attacker = _state([100, 0, 20])
    pursuers = np.tile(attacker, (3, 1))
    pursuers[0, 1] -= separation
    result = policy(attacker, pursuers)
    assert result.shape == (3,)
    assert np.isfinite(result).all()
    assert np.linalg.norm(result[:2]) <= policy.limits.max_horizontal_acceleration + 1e-12
    assert abs(result[2]) <= policy.limits.max_vertical_acceleration + 1e-12


def test_no_threat_rollout_continuously_approaches_base(policy, baseline_config):
    attacker = _state([499, 0, 31])
    pursuers = _pursuers([[-1000, -1000, 100], [-1000, 1000, 100], [-1000, 0, 100]])
    acceleration = np.zeros(3)
    previous_distance = np.linalg.norm(attacker[:3] - policy.base_position)
    for _ in range(150):
        attacker, acceleration = integrate(
            attacker, policy(attacker, pursuers), acceleration,
            baseline_config["environment"]["dt"], policy.limits,
        )
        distance = np.linalg.norm(attacker[:3] - policy.base_position)
        assert distance < previous_distance
        assert np.isfinite(attacker).all()
        previous_distance = distance


def test_optional_boundary_force_is_preserved_but_not_enabled_by_default(limits):
    policy = GoalDirectedAttackerPolicy(
        limits, 1e-6, base_position=[0, 0, 0], goal_weight=1, avoidance_weight=0,
        bounds_low=[-800, -800, -800], bounds_high=[800, 800, 800],
        boundary_margin_horizontal=50, boundary_margin_vertical=20, boundary_strength=2,
    )
    attacker = _state([775, 0, 0])
    assert policy.boundary_enabled
    np.testing.assert_allclose(policy.boundary_repulsion(attacker[:3]), [-1, 0, 0])
    np.testing.assert_allclose(policy(attacker, np.tile(attacker, (3, 1))), [-3, 0, 0])


@pytest.mark.parametrize("key,value", [
    ("goal_weight", 0), ("goal_weight", -1), ("goal_weight", np.nan),
    ("goal_weight", np.inf), ("goal_weight", 1j),
    ("avoidance_weight", -1), ("avoidance_weight", np.nan),
    ("avoidance_weight", np.inf), ("avoidance_weight", "bad"),
    ("base_position", [0, 0]), ("base_position", [0, 0, np.inf]),
    ("base_position", [0, 0, 1j]),
    ("distance_epsilon", 0), ("distance_epsilon", np.nan),
])
def test_invalid_policy_configuration_is_rejected(limits, key, value):
    parameters = dict(distance_epsilon=1e-6, base_position=[0, 0, 0],
                      goal_weight=1.0, avoidance_weight=25.0)
    parameters[key] = value
    with pytest.raises(ValueError):
        GoalDirectedAttackerPolicy(limits, **parameters)


@pytest.mark.parametrize("attacker,pursuers", [
    (np.zeros(3), np.zeros((3, 6))),
    (np.zeros(6), np.zeros((2, 6))),
    (np.array([0, 0, 0, 0, 0, np.nan]), np.zeros((3, 6))),
    (np.zeros(6), np.full((3, 6), np.inf)),
    (np.zeros(6, dtype=complex), np.zeros((3, 6))),
])
def test_invalid_aircraft_states_are_rejected(policy, attacker, pursuers):
    with pytest.raises(ValueError):
        policy(attacker, pursuers)
