"""Numerical contracts for replaying saved trajectories without an environment."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.offline_near_reward_replay import (
    category,
    discounted_return,
    near_shaping,
    reward_components,
)


COEFFICIENTS = {
    "distance_cost_coefficient": 0.01,
    "survival_penalty_per_second": -0.2,
    "capture_reward": 500.0,
    "attack_failure_penalty": -100.0,
}


def test_original_reward_uses_post_step_distance_and_preserves_inputs():
    distances = np.array([500.0, 100.0, 20.0, 4.0])
    previous_distances = distances.copy()
    coefficients = deepcopy(COEFFICIENTS)
    components = reward_components(distances, 0.1, True, False, coefficients)

    np.testing.assert_allclose(components["distance_cost"], [-0.1, -0.02, -0.004])
    np.testing.assert_allclose(components["survival"], [-0.02, -0.02, -0.02])
    np.testing.assert_array_equal(components["capture"], [0.0, 0.0, 500.0])
    np.testing.assert_array_equal(components["defense_failure"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(components["original"], [-0.12, -0.04, 499.976])
    for values in components.values():
        assert values.shape == (3,)
        assert np.isfinite(values).all()
    np.testing.assert_array_equal(distances, previous_distances)
    assert coefficients == COEFFICIENTS


@pytest.mark.parametrize("captured, failed, terminal", [(True, False, 500.0), (False, True, -100.0), (False, False, 0.0)])
def test_terminal_component_only_applies_on_last_step(captured, failed, terminal):
    components = reward_components([50.0, 40.0, 30.0], 0.1, captured, failed, COEFFICIENTS)
    np.testing.assert_array_equal(
        components["capture"] + components["defense_failure"], [0.0, terminal]
    )
    np.testing.assert_allclose(
        components["original"],
        components["distance_cost"] + components["survival"]
        + components["capture"] + components["defense_failure"],
    )


def test_capture_and_failure_cannot_both_receive_terminal_reward():
    with pytest.raises(ValueError):
        reward_components([10.0, 4.0], 0.1, True, True, COEFFICIENTS)


@pytest.mark.parametrize("length_scale_m", [10.0, 20.0, 30.0])
@pytest.mark.parametrize("beta", [20.0, 40.0, 60.0])
def test_near_shaping_matches_formula_and_telescopes(length_scale_m, beta):
    distances = np.array([500.0, 25.0, 8.0, 5.5, 25.0, 120.0])
    original = distances.copy()
    shaping = near_shaping(distances, length_scale_m, beta)
    potential = np.exp(-distances / length_scale_m)

    assert shaping.dtype == np.float64
    assert shaping.shape == (len(distances) - 1,)
    np.testing.assert_allclose(shaping, beta * np.diff(potential), rtol=1e-13, atol=1e-13)
    assert shaping[:3].min() > 0.0
    assert shaping[3:].max() < 0.0
    assert shaping.sum() == pytest.approx(beta * (potential[-1] - potential[0]), abs=1e-12)
    np.testing.assert_array_equal(distances, original)


def test_stationary_distance_and_zero_beta_cannot_accumulate_shaping():
    np.testing.assert_array_equal(near_shaping([6.0] * 101, 20.0, 40.0), np.zeros(100))
    np.testing.assert_array_equal(near_shaping([500.0, 5.0, 100.0], 20.0, 0.0), np.zeros(2))


def test_close_flyby_loses_positive_shaping_when_returning_to_same_distance():
    shaping = near_shaping([150.0, 30.0, 6.0, 30.0, 150.0], 20.0, 40.0)
    assert shaping[2] < 0.0
    assert shaping[3] < 0.0
    assert shaping.sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("gamma", [0.0, 0.99, 1.0])
def test_discounted_return_uses_first_reward_without_discount(gamma):
    rewards = np.array([-0.4, -0.2, 500.0])
    actual = discounted_return(rewards, gamma)
    assert actual == pytest.approx(-0.4 - 0.2 * gamma + 500.0 * gamma**2)


@pytest.mark.parametrize("gamma", [0.0, 0.99, 1.0])
def test_discounted_potential_difference_identity(gamma):
    distances = np.array([500.0, 50.0, 7.0, 25.0, 80.0])
    beta = 40.0
    potential = np.exp(-distances / 20.0)
    count = len(distances) - 1
    expected = beta * (
        -potential[0] + gamma ** (count - 1) * potential[-1]
        + (1.0 - gamma) * np.sum(gamma ** np.arange(count - 1) * potential[1:-1])
    )
    assert discounted_return(near_shaping(distances, 20.0, beta), gamma) == pytest.approx(expected)


def test_discounted_flyby_can_have_positive_contribution_despite_zero_total():
    shaping = near_shaping([100.0, 2.0, 2.0, 100.0], 20.0, 40.0)
    assert shaping.sum() == pytest.approx(0.0, abs=1e-12)
    assert discounted_return(shaping, 0.99) > 0.0


@pytest.mark.parametrize("captured, distance, expected", [
    (True, 4.0, "capture"),
    (False, 5.0, "near_miss"),
    (False, 10.0, "near_miss"),
    (False, 10.0001, "moderate_failure"),
    (False, 30.0, "moderate_failure"),
    (False, 30.0001, "other_failure"),
    (False, 100.0, "other_failure"),
    (False, 100.0001, "clear_failure"),
])
def test_category_boundaries(captured, distance, expected):
    assert category(captured, distance) == expected


@pytest.mark.parametrize("distances", [[], [1.0], [[2.0, 1.0]], [2.0, np.nan], [np.inf, 1.0], [2.0, -1.0]])
def test_invalid_distance_series_is_rejected(distances):
    with pytest.raises(ValueError):
        near_shaping(distances, 20.0, 40.0)
    with pytest.raises(ValueError):
        reward_components(distances, 0.1, False, False, COEFFICIENTS)


@pytest.mark.parametrize("length_scale_m", [0.0, -1.0, np.nan, np.inf])
def test_invalid_length_scale_is_rejected(length_scale_m):
    with pytest.raises(ValueError):
        near_shaping([10.0, 5.0], length_scale_m, 40.0)


@pytest.mark.parametrize("beta", [-1.0, np.nan, np.inf])
def test_invalid_beta_is_rejected(beta):
    with pytest.raises(ValueError):
        near_shaping([10.0, 5.0], 20.0, beta)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_invalid_time_step_is_rejected(dt):
    with pytest.raises(ValueError):
        reward_components([10.0, 5.0], dt, False, False, COEFFICIENTS)


@pytest.mark.parametrize("gamma", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_discount_is_rejected(gamma):
    with pytest.raises(ValueError):
        discounted_return([1.0, 2.0], gamma)


@pytest.mark.parametrize("rewards", [[], [[1.0, 2.0]], [1.0, np.nan], [1.0, np.inf]])
def test_invalid_reward_series_is_rejected(rewards):
    with pytest.raises(ValueError):
        discounted_return(rewards, 0.99)
