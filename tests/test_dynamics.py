"""Physical invariants of the point-mass action pipeline."""

import numpy as np
import pytest

from envs.dynamics import MotionLimits, integrate, scale_action


@pytest.fixture
def limits(baseline_config):
    return MotionLimits.from_config(baseline_config["pursuer"])


def test_normalized_actions_are_scaled_clipped_and_norm_limited(limits):
    actions = np.array([[0.5, 0.0, -0.5], [2.0, -2.0, 2.0]])
    acceleration = scale_action(actions, limits)
    np.testing.assert_allclose(acceleration[0], [1.5, 0.0, -1.0])
    np.testing.assert_allclose(
        acceleration[1], [3.0 / np.sqrt(2), -3.0 / np.sqrt(2), 2.0]
    )
    np.testing.assert_array_equal(actions[1], [2.0, -2.0, 2.0])
    assert acceleration.shape == (2, 3)


def test_scaling_uses_configured_limits(baseline_config):
    values = dict(baseline_config["pursuer"])
    values.update(max_horizontal_acceleration=7.0, max_vertical_acceleration=4.0)
    custom = MotionLimits.from_config(values)
    np.testing.assert_allclose(scale_action([1.0, 0.0, -1.0], custom), [7.0, 0.0, -4.0])


def test_position_update_uses_the_new_velocity(limits):
    state = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    next_state, acceleration = integrate(state, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.1, limits)
    np.testing.assert_allclose(acceleration, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(next_state, [1.01, 2.0, 3.0, 0.1, 0.0, 0.0])
    np.testing.assert_array_equal(state, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])


def test_integrator_limits_physical_acceleration(limits):
    next_state, acceleration = integrate(
        [0.0, 0.0, 50.0, 0.0, 0.0, 0.0], [100.0, 100.0, -100.0],
        np.zeros(3), 1.0, limits,
    )
    assert np.linalg.norm(acceleration[:2]) == pytest.approx(limits.max_horizontal_acceleration)
    assert acceleration[2] == pytest.approx(-limits.max_vertical_acceleration)
    assert np.isfinite(next_state).all()


@pytest.mark.parametrize("vertical_velocity,command,expected", [(7.9, 2.0, 8.0), (-5.9, -2.0, -6.0)])
def test_horizontal_and_asymmetric_vertical_speed_caps(limits, vertical_velocity, command, expected):
    state = [0.0, 0.0, 50.0, 19.0, 19.0, vertical_velocity]
    next_state, _ = integrate(state, [3.0, 3.0, command], [0.0, 0.0, 0.0], 1.0, limits)
    assert np.linalg.norm(next_state[3:5]) == pytest.approx(limits.max_horizontal_speed)
    assert next_state[5] == pytest.approx(expected)
    np.testing.assert_allclose(next_state[:3], np.array(state[:3]) + next_state[3:])


def test_jerk_limits_vector_change_across_command_reversals(limits):
    state = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 0.0])
    previous = np.zeros(3)
    dt = 0.1
    commands = ([3.0, 0.0, 2.0], [-3.0, 0.0, -2.0], [0.0, 3.0, 2.0])
    for step in range(60):
        state, applied = integrate(state, commands[(step // 10) % 3], previous, dt, limits)
        assert np.linalg.norm(applied - previous) <= limits.max_jerk * dt + 1e-12
        assert np.linalg.norm(applied[:2]) <= limits.max_horizontal_acceleration + 1e-12
        assert abs(applied[2]) <= limits.max_vertical_acceleration + 1e-12
        assert np.linalg.norm(state[3:5]) <= limits.max_horizontal_speed + 1e-12
        assert -limits.max_descent_speed <= state[5] <= limits.max_climb_speed
        assert np.isfinite(state).all()
        previous = applied.copy()
