"""Offline geometry, reward timing and stationary-replay contracts."""

from copy import deepcopy

import numpy as np
import pytest

from utils.line_cost_replay import (
    episode_components,
    line_cost,
    segment_distances,
    stationary_counterfactual,
)


@pytest.fixture
def environment():
    return {
        "environment": {"dt": 0.1, "max_episode_steps": 500},
        "mission": {"base_position": [0.0, 0.0, 0.0], "attack_radius": 10.0},
        "capture": {"radius": 5.0},
        "initialization": {
            "pursuer_position": [1.5, 0.0, 1.0],
            "pursuer_velocity": [0.0, 0.0, 0.0],
        },
        "attacker": {"policy": {"mode": "direct"}},
        "reward": {
            "distance_cost_coefficient": 0.01,
            "survival_penalty_per_second": -0.2,
            "capture_reward": 500.0,
            "attack_failure_penalty": -100.0,
            "best_so_far_shaping": {"length_scale_m": 20.0, "beta": 100.0},
        },
    }


def saved_states(attacker_x):
    states = np.zeros((len(attacker_x), 2, 6), dtype=np.float64)
    states[:, 0, :3] = [30.0, 40.0, 50.0]
    states[:, 0, 3:] = [2.0, 3.0, 4.0]
    states[:, 1, 0] = attacker_x
    states[:, 1, 3] = -20.0
    return states


def test_segment_distance_uses_three_dimensions():
    pursuer = np.array([[5.0, 3.0, 4.0], [5.0, -3.0, -4.0], [5.0, 0.0, 0.0]])
    distances = segment_distances(pursuer, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    np.testing.assert_allclose(distances, [5.0, 5.0, 0.0])
    assert distances.shape == (3,)
    assert distances.dtype == np.float64


def test_segment_endpoints_and_extension_are_distinct():
    positions = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        [-3.0, 4.0, 0.0], [13.0, 4.0, 0.0],
        [-3.0, 0.0, 0.0], [13.0, 0.0, 0.0],
    ])

    distances = segment_distances(positions, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    np.testing.assert_allclose(distances, [0.0, 0.0, 5.0, 5.0, 3.0, 3.0])


def test_coincident_attacker_and_base_reduce_to_point_distance():
    distances = segment_distances(
        [[4.0, 6.0, 3.0], [1.0, 2.0, 3.0]],
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
    )

    np.testing.assert_allclose(distances, [5.0, 0.0])
    assert np.isfinite(distances).all()


def test_batched_segments_may_degenerate_independently():
    distances = segment_distances(
        [[5.0, 3.0, 4.0], [0.0, 3.0, 4.0]],
        [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(distances, [5.0, 5.0])


def test_one_point_returns_a_scalar_distance():
    result = segment_distances([5, 3, 4], [10, 0, 0], [0, 0, 0])

    assert np.ndim(result) == 0
    assert result == pytest.approx(5.0)


def test_geometry_is_translation_and_rotation_invariant_and_preserves_inputs():
    pursuer = np.array([[5.0, 3.0, 4.0], [-2.0, 4.0, 5.0], [14.0, -3.0, 2.0]])
    attacker = np.array([[10.0, 0.0, 0.0], [8.0, -2.0, 4.0], [10.0, 0.0, 2.0]])
    base = np.array([0.0, 0.0, 0.0])
    originals = [array.copy() for array in (pursuer, attacker, base)]
    angle = np.deg2rad(37.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    offset = np.array([15.0, -23.0, 17.0])

    actual = segment_distances(pursuer, attacker, base)
    transformed = segment_distances(
        pursuer @ rotation.T + offset,
        attacker @ rotation.T + offset,
        base @ rotation.T + offset,
    )
    swapped = segment_distances(pursuer, base, attacker)

    np.testing.assert_allclose(transformed, actual, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(swapped, actual, rtol=1e-13, atol=1e-13)
    for array, original in zip((pursuer, attacker, base), originals):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("angle_degrees, expected", [
    (-30.0, 1.1449321730590807),
    (0.0, 0.8449874371066199),
    (30.0, 1.1449321730590807),
])
def test_current_pursuer_initial_position_is_already_near_the_segment(angle_degrees, expected):
    angle = np.deg2rad(angle_degrees)
    attacker = [497.49371855331 * np.cos(angle), 497.49371855331 * np.sin(angle), 50.0]

    actual = segment_distances([1.5, 0.0, 1.0], attacker, [0.0, 0.0, 0.0])

    assert actual == pytest.approx(expected, abs=1e-12)
    assert np.linalg.norm(np.asarray(attacker) - [1.5, 0.0, 1.0]) > 498.0


@pytest.mark.parametrize("argument", [
    [], 1.0, [1.0, 2.0], [1.0, 2.0, 3.0, 4.0],
    [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0],
])
@pytest.mark.parametrize("position_index", [0, 1, 2])
def test_geometry_rejects_invalid_position_arrays(argument, position_index):
    arguments = [[5.0, 3.0, 4.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    arguments[position_index] = argument

    with pytest.raises(ValueError):
        segment_distances(*arguments)


def test_geometry_rejects_incompatible_batches():
    with pytest.raises(ValueError):
        segment_distances(np.zeros((2, 3)), np.zeros((4, 3)), np.zeros(3))


def test_line_cost_uses_post_step_distance_and_actual_time_intervals():
    distances = np.array([100.0, 2.0, 4.0, 1.0])
    time = np.array([0.0, 0.1, 0.3, 0.6])
    original_distances, original_time = distances.copy(), time.copy()

    cost = line_cost(distances, time, 0.5)

    np.testing.assert_allclose(cost, [-0.1, -0.4, -0.15], atol=1e-14)
    assert cost.shape == (3,)
    assert cost.dtype == np.float64
    assert np.isfinite(cost).all()
    np.testing.assert_array_equal(distances, original_distances)
    np.testing.assert_array_equal(time, original_time)


def test_cost_scales_linearly_with_coefficient_and_elapsed_time():
    distance = [5.0, 5.0, 5.0]
    ordinary = line_cost(distance, [0.0, 0.1, 0.2], 0.02)

    np.testing.assert_allclose(line_cost(distance, [0.0, 0.1, 0.2], 0.04), 2 * ordinary)
    np.testing.assert_allclose(line_cost(distance, [0.0, 0.2, 0.4], 0.02), 2 * ordinary)
    np.testing.assert_allclose(line_cost(distance, [7.0, 7.1, 7.2], 0.02), ordinary)


def test_zero_coefficient_and_perfect_alignment_have_zero_cost():
    np.testing.assert_array_equal(line_cost([10.0, 8.0, 6.0], [0.0, 0.1, 0.2], 0.0), np.zeros(2))
    np.testing.assert_array_equal(line_cost([0.0, 0.0, 0.0], [0.0, 0.1, 0.2], 0.1), np.zeros(2))


def test_stationary_base_guard_cannot_earn_positive_line_reward():
    distances = segment_distances(
        np.repeat([[1.5, 0.0, 1.0]], 4, axis=0),
        [[497.49371855331, 0.0, 50.0], [200.0, 0.0, 20.0],
         [20.0, 0.0, 2.0], [8.0, 0.0, 0.8]],
        [0.0, 0.0, 0.0],
    )

    cost = line_cost(distances, [0.0, 0.1, 0.2, 0.3], 0.02)

    assert np.all(cost <= 0.0)
    assert cost.sum() < 0.0


@pytest.mark.parametrize("distances, time", [
    ([], []), ([1.0], [0.0]), ([[1.0, 2.0]], [0.0, 0.1]),
    ([1.0, 2.0], [0.0]), ([1.0, -1.0], [0.0, 0.1]),
    ([1.0, np.nan], [0.0, 0.1]), ([np.inf, 2.0], [0.0, 0.1]),
    ([1.0, 2.0], [0.0, np.nan]), ([1.0, 2.0], [0.0, np.inf]),
    ([1.0, 2.0], [0.0, 0.0]), ([1.0, 2.0], [0.1, 0.0]),
])
def test_line_cost_rejects_invalid_distances_or_timestamps(distances, time):
    with pytest.raises(ValueError):
        line_cost(distances, time, 0.01)


@pytest.mark.parametrize("coefficient", [-0.01, np.nan, np.inf, [0.01, 0.02]])
def test_line_cost_rejects_invalid_coefficients(coefficient):
    with pytest.raises(ValueError):
        line_cost([1.0, 2.0], [0.0, 0.1], coefficient)


def test_stationary_replay_stops_at_first_failure_and_preserves_saved_inputs(environment):
    states = saved_states([20.0, 15.0, 9.0, 1.0])
    time = np.arange(4) * 0.1
    original_states, original_time = states.copy(), time.copy()
    original_environment = deepcopy(environment)

    result = stationary_counterfactual(states, time, environment)

    assert result["event_index"] == 2
    assert result["defense_failure"] is True
    assert result["captured"] is False
    assert result["timeout"] is False
    assert result["states"].shape == (3, 2, 6)
    np.testing.assert_array_equal(result["states"][:, 0, :3], np.repeat([[1.5, 0.0, 1.0]], 3, axis=0))
    np.testing.assert_array_equal(result["states"][:, 0, 3:], np.zeros((3, 3)))
    np.testing.assert_array_equal(result["states"][:, 1], states[:3, 1])
    np.testing.assert_array_equal(result["time"], time[:3])
    np.testing.assert_array_equal(states, original_states)
    np.testing.assert_array_equal(time, original_time)
    assert environment == original_environment
    np.testing.assert_array_equal(result["components"]["defense_failure"], [0.0, -100.0])
    np.testing.assert_array_equal(result["components"]["capture"], [0.0, 0.0])
    np.testing.assert_allclose(
        result["components"]["total"],
        result["components"]["original"] + result["components"]["near_shaping"],
    )


def test_reconstructed_existing_reward_matches_independent_original_and_shaping_formula(environment):
    states = saved_states([20.0, 11.0, 10.0])
    states[:, 0, :3] = [6.0, 0.0, 0.0]
    states[:, 0, 3:] = 0.0
    expected_distances = np.array([14.0, 5.0, 4.0])
    expected_original = np.array([-0.025, 499.976])
    expected_shaping = 100.0 * np.diff(np.exp(-expected_distances / 20.0))

    result = episode_components(states, [0.0, 0.1, 0.2], environment)

    np.testing.assert_allclose(result["distances"], expected_distances, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result["components"]["distance_cost"], [-0.005, -0.004], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result["components"]["survival"], [-0.02, -0.02], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result["components"]["original"], expected_original, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result["components"]["near_shaping"], expected_shaping, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        result["components"]["total"], expected_original + expected_shaping,
        rtol=0.0, atol=1e-12,
    )


def test_added_line_cost_reconciles_discounted_return_without_changing_terminal_terms(environment):
    states = saved_states([20.0, 11.0, 10.0])
    states[:, 0, :3] = [6.0, 2.0, 0.0]
    states[:, 0, 3:] = 0.0
    time = np.array([0.0, 0.1, 0.2])
    result = episode_components(states, time, environment)
    original_capture = result["components"]["capture"].copy()
    original_failure = result["components"]["defense_failure"].copy()
    offset = segment_distances(states[:, 0, :3], states[:, 1, :3], [0.0, 0.0, 0.0])
    extra = line_cost(offset, time, 0.02)
    discount = 0.99 ** np.arange(2)
    total = result["components"]["total"]

    np.testing.assert_allclose(extra, [-0.004, -0.004], rtol=0.0, atol=1e-14)
    assert np.dot(total + extra, discount) == pytest.approx(
        np.dot(total, discount) + np.dot(extra, discount), abs=1e-12,
    )
    np.testing.assert_array_equal(result["components"]["capture"], original_capture)
    np.testing.assert_array_equal(result["components"]["defense_failure"], original_failure)


def test_stationary_capture_has_priority_over_simultaneous_base_entry(environment):
    environment["initialization"]["pursuer_position"] = [6.0, 0.0, 0.0]

    result = stationary_counterfactual(saved_states([20.0, 9.0, 0.0]), [0.0, 0.1, 0.2], environment)

    assert result["captured"] is True
    assert result["defense_failure"] is False
    assert result["event_index"] == 1
    np.testing.assert_array_equal(result["components"]["capture"], [500.0])
    np.testing.assert_array_equal(result["components"]["defense_failure"], [0.0])


def test_stationary_capture_radius_is_strict(environment):
    environment["initialization"]["pursuer_position"] = [6.0, 0.0, 0.0]

    result = stationary_counterfactual(saved_states([20.0, 11.0, 10.0]), [0.0, 0.1, 0.2], environment)

    assert result["event_index"] == 2
    assert result["captured"] is True
    np.testing.assert_array_equal(result["components"]["capture"], [0.0, 500.0])


def test_stationary_failure_radius_is_strict(environment):
    environment["initialization"]["pursuer_position"] = [-100.0, 0.0, 0.0]

    result = stationary_counterfactual(saved_states([20.0, 10.0, 9.0]), [0.0, 0.1, 0.2], environment)

    assert result["event_index"] == 2
    assert result["defense_failure"] is True
    np.testing.assert_array_equal(result["components"]["defense_failure"], [0.0, -100.0])


def test_stationary_replay_honors_timeout(environment):
    environment["environment"]["max_episode_steps"] = 2
    environment["initialization"]["pursuer_position"] = [-100.0, 0.0, 0.0]

    result = stationary_counterfactual(saved_states([30.0, 25.0, 20.0, 15.0]), np.arange(4) * 0.1, environment)

    assert result["event_index"] == 2
    assert result["timeout"] is True
    assert result["captured"] is False
    assert result["defense_failure"] is False
    np.testing.assert_array_equal(result["components"]["capture"], np.zeros(2))
    np.testing.assert_array_equal(result["components"]["defense_failure"], np.zeros(2))


def test_counterfactual_requires_attacker_path_through_a_terminal_event(environment):
    with pytest.raises(ValueError, match="terminal event"):
        stationary_counterfactual(saved_states([100.0, 90.0]), [0.0, 0.1], environment)


def test_counterfactual_requires_independent_direct_attacker(environment):
    environment["attacker"]["policy"]["mode"] = "repulsion"

    with pytest.raises(ValueError, match="Direct Attacker"):
        stationary_counterfactual(saved_states([20.0, 9.0]), [0.0, 0.1], environment)


def test_counterfactual_requires_initial_zero_velocity(environment):
    environment["initialization"]["pursuer_velocity"] = [0.1, 0.0, 0.0]

    with pytest.raises(ValueError, match="zero-velocity"):
        stationary_counterfactual(saved_states([20.0, 9.0]), [0.0, 0.1], environment)


def test_existing_components_require_exactly_one_completed_episode(environment):
    states = saved_states([20.0, 9.0, 1.0])
    states[:, 0, :3] = environment["initialization"]["pursuer_position"]
    states[:, 0, 3:] = 0.0

    with pytest.raises(ValueError, match="first terminal"):
        episode_components(states, [0.0, 0.1, 0.2], environment)
