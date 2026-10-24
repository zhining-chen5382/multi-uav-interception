"""Rule-based evasion direction and numerical edge cases."""

import numpy as np
import pytest

from envs.dynamics import MotionLimits
from envs.evader_policy import RepulsiveEvaderPolicy


@pytest.fixture
def policy(baseline_config):
    limits = MotionLimits.from_config(baseline_config["evader"])
    return RepulsiveEvaderPolicy(limits, baseline_config["evader"]["policy"]["distance_epsilon"])


def test_evader_accelerates_away_from_pursuers(policy):
    evader = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 0.0])
    pursuers = np.array([[-10.0, 0.0, 50.0, 0.0, 0.0, 0.0]] * 3)
    acceleration = policy(evader, pursuers)
    np.testing.assert_allclose(acceleration, [3.0, 0.0, 0.0])
    np.testing.assert_array_equal(acceleration, policy(evader, pursuers))


def test_nearby_pursuer_dominates_opposite_distant_pursuers(policy):
    evader = np.zeros(6)
    pursuers = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [-20.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert policy(evader, pursuers)[0] < 0.0


def test_symmetric_and_coincident_states_have_no_nan(policy):
    evader = np.zeros(6)
    symmetric = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(policy(evader, symmetric), np.zeros(3))
    coincident = policy(evader, np.zeros((3, 6)))
    assert coincident.shape == (3,)
    assert np.isfinite(coincident).all()
    np.testing.assert_allclose(coincident, np.zeros(3))


def test_policy_obeys_custom_acceleration_parameters(baseline_config):
    values = dict(baseline_config["evader"])
    values.update(max_horizontal_acceleration=1.2, max_vertical_acceleration=0.7)
    policy = RepulsiveEvaderPolicy(MotionLimits.from_config(values), 1e-6)
    pursuers = np.array([[-1.0, 0.0, -1.0, 0.0, 0.0, 0.0]] * 3)
    acceleration = policy(np.zeros(6), pursuers)
    np.testing.assert_allclose(acceleration, [1.2 / np.sqrt(2), 0.0, 0.7 / np.sqrt(2)])
