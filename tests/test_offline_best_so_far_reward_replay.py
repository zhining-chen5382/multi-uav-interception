"""Numerical checks for rewarding only new best-so-far distance records."""

import numpy as np
import pytest

from scripts.offline_best_so_far_reward_replay import best_so_far_shaping
from scripts.offline_near_reward_replay import discounted_return, near_shaping


@pytest.mark.parametrize("length_scale_m", [10.0, 20.0, 30.0])
@pytest.mark.parametrize("beta", [20.0, 40.0, 60.0])
def test_nine_parameter_grid_matches_formula_and_telescopes(length_scale_m, beta):
    distances = np.array([500.0, 25.0, 8.0, 25.0, 8.0, 6.0, 120.0])
    before = distances.copy()
    best_distances = np.array([500.0, 25.0, 8.0, 8.0, 8.0, 6.0, 6.0])
    potential = np.exp(-best_distances / length_scale_m)

    shaping = best_so_far_shaping(distances, length_scale_m, beta)

    assert shaping.shape == (len(distances) - 1,)
    assert shaping.dtype == np.float64
    assert np.isfinite(shaping).all()
    assert np.all(shaping >= 0.0)
    np.testing.assert_allclose(shaping, beta * np.diff(potential), rtol=1e-13, atol=1e-13)
    assert shaping.sum() == pytest.approx(
        beta * (np.exp(-distances.min() / length_scale_m)
                - np.exp(-distances[0] / length_scale_m)),
        abs=1e-12,
    )
    np.testing.assert_array_equal(distances, before)


def test_flyby_retains_the_reward_when_distance_increases():
    distances = [150.0, 30.0, 6.0, 30.0, 150.0]
    shaping = best_so_far_shaping(distances, 20.0, 40.0)

    assert shaping[0] > 0.0
    assert shaping[1] > 0.0
    np.testing.assert_array_equal(shaping[2:], [0.0, 0.0])
    assert shaping.sum() == pytest.approx(
        40.0 * (np.exp(-6.0 / 20.0) - np.exp(-150.0 / 20.0))
    )
    assert near_shaping(distances, 20.0, 40.0).sum() == pytest.approx(0.0, abs=1e-12)


def test_approaching_again_requires_breaking_the_previous_record():
    distances = [100.0, 10.0, 40.0, 25.0, 10.0, 8.0, 9.0, 8.0, 6.0]
    shaping = best_so_far_shaping(distances, 20.0, 40.0)

    np.testing.assert_array_equal(shaping[[1, 2, 3, 5, 6]], np.zeros(5))
    assert shaping[0] > 0.0
    assert shaping[4] == pytest.approx(40.0 * (np.exp(-8.0 / 20.0) - np.exp(-10.0 / 20.0)))
    assert shaping[7] == pytest.approx(40.0 * (np.exp(-6.0 / 20.0) - np.exp(-8.0 / 20.0)))


def test_stationary_and_retreating_distances_award_zero():
    np.testing.assert_array_equal(best_so_far_shaping([6.0] * 101, 20.0, 40.0), np.zeros(100))
    np.testing.assert_array_equal(best_so_far_shaping([6.0, 10.0, 30.0, 100.0], 20.0, 40.0), np.zeros(3))


def test_repeated_same_distance_segment_cannot_be_rewarded_again():
    one_approach = best_so_far_shaping([100.0, 30.0, 6.0], 20.0, 40.0)
    repeated = best_so_far_shaping([100.0, 30.0, 6.0] + [30.0, 100.0, 30.0, 6.0] * 25, 20.0, 40.0)

    np.testing.assert_array_equal(repeated[:2], one_approach)
    np.testing.assert_array_equal(repeated[2:], np.zeros(100))
    assert repeated.sum() == pytest.approx(one_approach.sum(), abs=1e-12)


def test_new_records_award_only_the_previously_unrewarded_interval():
    staged = best_so_far_shaping([100.0, 10.0, 40.0, 8.0, 40.0, 6.0], 20.0, 40.0)
    direct = best_so_far_shaping([100.0, 6.0], 20.0, 40.0)

    assert staged.sum() == pytest.approx(direct.sum(), abs=1e-12)
    assert staged[2] == pytest.approx(40.0 * (np.exp(-8.0 / 20.0) - np.exp(-10.0 / 20.0)))
    assert staged[4] == pytest.approx(40.0 * (np.exp(-6.0 / 20.0) - np.exp(-8.0 / 20.0)))


@pytest.mark.parametrize("length_scale_m", [10.0, 20.0, 30.0])
@pytest.mark.parametrize("beta", [20.0, 40.0, 60.0])
def test_episode_shaping_is_bounded_by_beta_and_below_failure_penalty(length_scale_m, beta):
    capture = best_so_far_shaping([500.0, 100.0, 6.0, 0.0], length_scale_m, beta)
    failure = best_so_far_shaping([500.0, 100.0, 5.0, 100.0], length_scale_m, beta)

    assert 0.0 <= capture.sum() <= beta
    assert 0.0 <= failure.sum() <= beta * np.exp(-5.0 / length_scale_m) < 100.0


def test_discounted_shaping_is_a_direct_sum_at_the_original_times():
    distances = [100.0, 6.0, 30.0, 6.0, 5.0, 100.0]
    shaping = best_so_far_shaping(distances, 20.0, 40.0)
    gamma = 0.99
    first_record = 40.0 * (np.exp(-6.0 / 20.0) - np.exp(-100.0 / 20.0))
    second_record = 40.0 * (np.exp(-5.0 / 20.0) - np.exp(-6.0 / 20.0))

    np.testing.assert_allclose(shaping, [first_record, 0.0, 0.0, second_record, 0.0])
    assert discounted_return(shaping, gamma) == pytest.approx(first_record + gamma**3 * second_record)
    assert discounted_return(shaping, gamma) > discounted_return(near_shaping(distances, 20.0, 40.0), gamma)


@pytest.mark.parametrize("gamma", [0.0, 0.99, 1.0])
def test_discounted_best_potential_telescoping_identity(gamma):
    distances = np.array([500.0, 50.0, 7.0, 25.0, 6.0, 80.0])
    potential = np.exp(-np.minimum.accumulate(distances) / 20.0)
    count = len(distances) - 1
    expected = 40.0 * (
        -potential[0] + gamma ** (count - 1) * potential[-1]
        + (1.0 - gamma) * np.sum(gamma ** np.arange(count - 1) * potential[1:-1])
    )

    assert discounted_return(best_so_far_shaping(distances, 20.0, 40.0), gamma) == pytest.approx(expected)


def test_monotonic_approach_matches_previous_distance_difference_version():
    distances = [500.0, 100.0, 20.0, 7.0, 4.0]
    np.testing.assert_array_equal(
        best_so_far_shaping(distances, 20.0, 40.0),
        near_shaping(distances, 20.0, 40.0),
    )


def test_zero_beta_has_no_reward_for_any_record():
    np.testing.assert_array_equal(best_so_far_shaping([100.0, 5.0, 100.0, 4.0], 20.0, 0.0), np.zeros(3))


def test_integer_distances_are_converted_to_finite_float64():
    shaping = best_so_far_shaping(np.array([100, 5, 100, 4]), 20, 40)
    assert shaping.dtype == np.float64
    assert np.isfinite(shaping).all()
    assert shaping[1] == 0.0


@pytest.mark.parametrize("distances", [
    [], [1.0], [[2.0, 1.0]], [2.0, np.nan], [np.inf, 1.0],
    [2.0, np.inf, 1.0], [2.0, 1.0, np.inf], [2.0, -1.0],
])
def test_invalid_original_distance_series_is_rejected(distances):
    with pytest.raises(ValueError):
        best_so_far_shaping(distances, 20.0, 40.0)


@pytest.mark.parametrize("length_scale_m", [0.0, -1.0, np.nan, np.inf])
def test_invalid_length_scale_is_rejected(length_scale_m):
    with pytest.raises(ValueError):
        best_so_far_shaping([10.0, 5.0], length_scale_m, 40.0)


@pytest.mark.parametrize("beta", [-1.0, np.nan, np.inf])
def test_invalid_beta_is_rejected(beta):
    with pytest.raises(ValueError):
        best_so_far_shaping([10.0, 5.0], 20.0, beta)
