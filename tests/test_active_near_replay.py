"""Contracts for the offline, position-attributed near shaping candidate."""

import numpy as np
import pytest

from scripts.offline_best_so_far_reward_replay import best_so_far_shaping
from utils.active_near_replay import active_near_shaping


def trajectory(pursuer_positions, attacker_positions):
    pursuer = np.asarray(pursuer_positions, dtype=np.float64)
    attacker = np.asarray(attacker_positions, dtype=np.float64)
    states = np.zeros((len(pursuer), 2, 6), dtype=np.float64)
    states[:, 0, :3] = pursuer
    states[:, 1, :3] = attacker
    return states


def collinear(pursuer_x, attacker_x):
    pursuer = np.zeros((len(pursuer_x), 3))
    attacker = np.zeros((len(attacker_x), 3))
    pursuer[:, 0] = pursuer_x
    attacker[:, 0] = attacker_x
    return trajectory(pursuer, attacker)


@pytest.mark.parametrize("own_displacement", [0.0, 1.0, -1.0])
def test_one_step_counterfactual_matches_analytic_example(own_displacement):
    states = collinear([0.0, own_displacement], [20.0, 18.0])
    result = active_near_shaping(states, 20.0, 100.0)
    final_distance = 18.0 - own_displacement
    original = 100.0 * (np.exp(-final_distance / 20.0) - np.exp(-1.0))
    corrected = 100.0 * max(0.0, np.exp(-final_distance / 20.0) - np.exp(-0.9))

    np.testing.assert_allclose(result["distances"], [20.0, final_distance])
    np.testing.assert_allclose(result["counterfactual_distances"], [18.0])
    np.testing.assert_allclose(result["counterfactual_best_distances"], [18.0])
    np.testing.assert_allclose(result["original_near"], [original], atol=1e-13)
    np.testing.assert_allclose(result["active_near"], [corrected], atol=1e-13)
    np.testing.assert_allclose(result["removed_near"], [original - corrected], atol=1e-13)


def test_stationary_pursuer_receives_zero_for_arbitrary_attacker_motion():
    pursuer = np.tile([1.5, -2.0, 3.0], (7, 1))
    attacker = [[100.0, 3.0, 8.0], [80.0, -5.0, 7.0], [30.0, 1.0, 3.0],
                [40.0, -2.0, 3.0], [4.0, 1.0, 3.0], [20.0, -3.0, 7.0],
                [1.5, -2.0, 3.0]]
    result = active_near_shaping(trajectory(pursuer, attacker), 20.0, 100.0)

    assert result["original_near"].sum() > 0
    np.testing.assert_array_equal(result["active_near"], np.zeros(6))
    np.testing.assert_array_equal(result["removed_near"], result["original_near"])


def test_unbroken_history_record_is_required_for_reward():
    states = collinear([0.0, -10.0, -5.0, 0.0], [10.0, 10.0, 10.0, 10.0])
    result = active_near_shaping(states, 20.0, 100.0)

    np.testing.assert_array_equal(result["best_distances"], [10.0] * 4)
    np.testing.assert_array_equal(result["active_near"], np.zeros(3))


def test_revisiting_distance_does_not_reissue_reward():
    states = collinear([0.0, 5.0, 0.0, 5.0, 6.0], [20.0] * 5)
    result = active_near_shaping(states, 20.0, 100.0)

    assert result["active_near"][0] > 0
    np.testing.assert_array_equal(result["active_near"][1:3], [0.0, 0.0])
    assert result["active_near"][3] > 0
    np.testing.assert_allclose(result["active_near"], result["original_near"])
    expected = 100.0 * (np.exp(-14.0 / 20.0) - np.exp(-20.0 / 20.0))
    assert result["active_near"].sum() == pytest.approx(expected)


def test_passive_record_consumes_history_before_later_active_motion():
    states = collinear([0.0, 0.0, -5.0, 0.0, 1.0], [20.0, 10.0, 10.0, 10.0, 10.0])
    result = active_near_shaping(states, 20.0, 100.0)

    np.testing.assert_array_equal(result["active_near"][:3], [0.0, 0.0, 0.0])
    assert result["active_near"][3] > 0
    assert result["removed_near"][0] > 0


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_actual_own_displacement_counts_in_every_spatial_axis(axis):
    pursuer = np.zeros((2, 3))
    pursuer[1, axis] = 1.0
    attacker = np.zeros((2, 3))
    attacker[:, axis] = [20.0, 18.0]
    result = active_near_shaping(trajectory(pursuer, attacker), 20.0, 100.0)

    assert result["active_near"][0] == pytest.approx(2.0845272208127597)


def test_inertial_motion_counts_without_an_acceleration_input():
    states = collinear([0.0, 1.0, 2.0], [20.0, 18.0, 16.0])
    states[:, 0, 3] = 10.0
    result = active_near_shaping(states, 20.0, 100.0)

    assert np.all(result["active_near"] > 0)
    changed_velocity = states.copy()
    changed_velocity[:, :, 3:] = 0
    comparison = active_near_shaping(changed_velocity, 20.0, 100.0)
    np.testing.assert_array_equal(result["active_near"], comparison["active_near"])


def test_shaping_is_rigid_transform_invariant_and_preserves_input():
    rng = np.random.RandomState(12)
    states = np.cumsum(rng.normal(size=(32, 2, 6)), axis=0)
    saved = states.copy()
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed = states.copy()
    transformed[:, :, :3] = states[:, :, :3] @ rotation + [35.0, -17.0, 8.0]
    expected = active_near_shaping(states, 20.0, 100.0)
    actual = active_near_shaping(transformed, 20.0, 100.0)

    np.testing.assert_array_equal(states, saved)
    for field in expected:
        np.testing.assert_allclose(actual[field], expected[field], atol=1e-12, rtol=1e-12)
        assert expected[field].dtype == np.float64
        assert np.isfinite(expected[field]).all()
        expected_length = len(states) if field in ("distances", "best_distances") else len(states) - 1
        assert expected[field].shape == (expected_length,)


def test_stepwise_and_total_bounds_match_existing_shaping():
    rng = np.random.RandomState(720)
    for _ in range(20):
        states = np.cumsum(rng.normal(size=(121, 2, 6)), axis=0)
        states[:, 1, 0] += 30.0
        result = active_near_shaping(states, 20.0, 100.0)
        original = best_so_far_shaping(result["distances"], 20.0, 100.0)

        np.testing.assert_allclose(result["original_near"], original, atol=1e-12, rtol=0)
        assert np.all(result["active_near"] >= 0)
        assert np.all(result["active_near"] <= original)
        assert np.all(result["removed_near"] >= 0)
        np.testing.assert_allclose(result["active_near"] + result["removed_near"], original)
        assert result["active_near"].sum() <= original.sum() + 1e-12
        assert original.sum() <= 100.0


def test_zero_beta_and_zero_distance_are_finite():
    states = np.zeros((3, 2, 6))
    for beta in (0.0, 100.0):
        result = active_near_shaping(states, 20.0, beta)
        for field in ("original_near", "active_near", "removed_near"):
            np.testing.assert_array_equal(result[field], [0.0, 0.0])


@pytest.mark.parametrize("shape", [(2, 6), (1, 2, 6), (0, 2, 6), (3, 3, 6), (3, 2, 5), (3, 2, 6, 1)])
def test_invalid_state_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="trajectory"):
        active_near_shaping(np.zeros(shape), 20.0, 100.0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_state_fields_are_rejected(value):
    states = np.zeros((3, 2, 6))
    states[1, 0, 5] = value
    with pytest.raises(ValueError, match="finite"):
        active_near_shaping(states, 20.0, 100.0)


@pytest.mark.parametrize("length_scale,beta", [
    (0.0, 100.0), (-20.0, 100.0), (np.nan, 100.0), (np.inf, 100.0),
    ([20.0], 100.0), (20.0, -1.0), (20.0, np.nan), (20.0, np.inf), (20.0, [100.0]),
])
def test_invalid_parameters_are_rejected(length_scale, beta):
    with pytest.raises(ValueError, match="scalars"):
        active_near_shaping(np.zeros((3, 2, 6)), length_scale, beta)


def test_finite_states_whose_distances_overflow_are_rejected():
    states = np.zeros((3, 2, 6))
    states[:, 0, 0] = -1e308
    states[:, 1, 0] = 1e308
    with pytest.raises(ValueError, match="distances"):
        active_near_shaping(states, 20.0, 100.0)


def test_small_positive_scale_produces_finite_saturated_potential():
    result = active_near_shaping(collinear([0.0, 1.0], [20.0, 18.0]), 1e-308, 100.0)
    np.testing.assert_array_equal(result["active_near"], [0.0])
    assert all(np.isfinite(value).all() for value in result.values())
