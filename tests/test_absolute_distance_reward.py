"""Behavior and compatibility checks for the opt-in distance-time cost."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from envs import PursuitEnv
from envs.rewards import absolute_distance_components


WEIGHTED_TERMS = (
    "absolute_distance_cost", "survival", "collision", "capture", "attack_failure",
)


@pytest.fixture
def distance_config(baseline_config):
    config = deepcopy(baseline_config)
    del config["reward"]["active_progress_coefficient"]
    config["reward"]["distance_cost_coefficient"] = 0.031
    return config


def _environment(tmp_path, config, name="distance"):
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return PursuitEnv(path)


def test_distance_cost_is_negative_and_prefers_remaining_close(distance_config):
    coefficients = distance_config["reward"]
    far = absolute_distance_components(50.0, 0.2, coefficients)
    near = absolute_distance_components(5.0, 0.2, coefficients)
    touching = absolute_distance_components(0.0, 0.2, coefficients)
    assert far["absolute_distance_cost"] == pytest.approx(-0.031 * 50.0 * 0.2)
    assert far["absolute_distance_cost"] < near["absolute_distance_cost"] < 0
    assert touching["absolute_distance_cost"] == 0.0
    assert sum(near[name] for name in WEIGHTED_TERMS) > sum(far[name] for name in WEIGHTED_TERMS)
    assert far["distance_to_nearest_pursuer"] == 50.0
    assert far["distance_time_integral"] == pytest.approx(10.0)
    # Equal terminal distance does not erase a previous interval spent far away.
    staying_near = 2 * near["absolute_distance_cost"]
    approaching_late = far["absolute_distance_cost"] + near["absolute_distance_cost"]
    assert staying_near > approaching_late


def test_same_distance_and_elapsed_time_have_same_cost_across_dt(distance_config):
    totals = []
    for dt in (0.05, 0.1, 0.2):
        parts = absolute_distance_components(45.0, dt, distance_config["reward"])
        steps = round(2.0 / dt)
        totals.append((parts["absolute_distance_cost"] * steps,
                       parts["survival"] * steps,
                       parts["distance_time_integral"] * steps))
    np.testing.assert_allclose(totals, np.tile([-0.031 * 45 * 2, -0.2 * 2, 45 * 2], (3, 1)))


def test_yaml_coefficient_changes_only_the_post_step_distance_cost(tmp_path, distance_config):
    other = deepcopy(distance_config)
    other["reward"]["distance_cost_coefficient"] *= 3
    first = _environment(tmp_path, distance_config, "first")
    second = _environment(tmp_path, other, "second")
    first.reset(seed=0)
    second.reset(seed=0)
    result_a = first.step(np.zeros((3, 3)))
    result_b = second.step(np.zeros((3, 3)))
    np.testing.assert_array_equal(first.get_state(), second.get_state())
    distance = result_a[4]["min_distance"]
    expected_difference = -2 * distance_config["reward"]["distance_cost_coefficient"] * distance * first.dt
    np.testing.assert_allclose(result_b[1] - result_a[1], np.full(3, expected_difference))
    assert result_b[4]["reward_components"]["absolute_distance_cost"] == pytest.approx(
        3 * result_a[4]["reward_components"]["absolute_distance_cost"]
    )


def test_weighted_terms_are_counted_once_in_shared_team_reward(tmp_path, distance_config):
    env = _environment(tmp_path, distance_config)
    _, before = env.reset(seed=4)
    observations, rewards, _, _, info = env.step(np.zeros((3, 3)))
    parts = info["reward_components"]
    assert info["min_distance"] != before["min_distance"]
    assert parts["distance_to_nearest_pursuer"] == info["min_distance"]
    assert parts["distance_time_integral"] == pytest.approx(info["min_distance"] * env.dt)
    assert parts["absolute_distance_cost"] == pytest.approx(
        -distance_config["reward"]["distance_cost_coefficient"] * info["min_distance"] * env.dt
    )
    team_reward = sum(parts[name] for name in WEIGHTED_TERMS)
    np.testing.assert_array_equal(rewards, np.full(3, team_reward))
    assert team_reward != pytest.approx(sum(parts.values()))
    assert set(info["episode_reward_components"]) == set(WEIGHTED_TERMS)
    assert observations.shape == (3, 21) and info["global_state"].shape == (24,)


@pytest.mark.parametrize("capture_at_base", [False, True])
def test_terminal_scale_and_capture_priority_are_unchanged(tmp_path, distance_config, capture_at_base):
    config = deepcopy(distance_config)
    config["initialization"]["attacker_position"] = [8.0, 0.0, 0.0]
    config["initialization"]["pursuer_positions"] = (
        [[8.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0]]
        if capture_at_base else [[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [-30.0, 0.0, 0.0]]
    )
    env = _environment(tmp_path, config)
    env.reset(seed=0)
    _, rewards, terminated, truncated, info = env.step(np.zeros((3, 3)))
    parts = info["reward_components"]
    assert terminated and not truncated and info["attacker_in_attack_region"]
    assert info["captured"] == capture_at_base
    assert info["enemy_attack_success"] == (not capture_at_base)
    assert parts["capture"] == (100.0 if capture_at_base else 0.0)
    assert parts["attack_failure"] == (0.0 if capture_at_base else -100.0)
    np.testing.assert_allclose(rewards, np.full(3, sum(parts[name] for name in WEIGHTED_TERMS)))
    # The arithmetic helper also enforces capture priority if both flags arrive.
    simultaneous = absolute_distance_components(1.0, env.dt, config["reward"],
                                                captured=True, attacker_reached_base=True)
    assert simultaneous["capture"] == 100.0 and simultaneous["attack_failure"] == 0.0


def test_three_unsafe_pairs_still_charge_one_team_collision_penalty(tmp_path, distance_config):
    config = deepcopy(distance_config)
    config["initialization"]["pursuer_positions"] = [[0.0, 0.0, 1.0]] * 3
    env = _environment(tmp_path, config)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros((3, 3)))
    assert info["collision_pairs"] == 3
    assert info["reward_components"]["collision"] == config["reward"]["collision_penalty"] == -5.0


def test_episode_totals_are_independent_copies_and_reset_between_episodes(tmp_path, distance_config):
    config = deepcopy(distance_config)
    config["environment"]["max_episode_steps"] = 3
    env = _environment(tmp_path, config)
    for seed in (4, 5):
        _, initial = env.reset(seed=seed)
        assert initial["episode_reward_components"] == dict.fromkeys(WEIGHTED_TERMS, 0.0)
        expected = dict.fromkeys(WEIGHTED_TERMS, 0.0)
        total = 0.0
        for step in range(3):
            _, rewards, terminated, truncated, info = env.step(env.sample_actions())
            total += rewards[0]
            for name in WEIGHTED_TERMS:
                expected[name] += info["reward_components"][name]
            assert info["episode_reward_components"] == pytest.approx(expected)
            if step == 0:
                info["episode_reward_components"]["absolute_distance_cost"] = 1e9
        assert truncated and not terminated
        assert sum(expected.values()) == pytest.approx(total)
        assert initial["episode_reward_components"] == dict.fromkeys(WEIGHTED_TERMS, 0.0)


@pytest.mark.parametrize("distance,dt", [
    (-1.0, 0.1), (np.nan, 0.1), (np.inf, 0.1),
    (1.0, 0.0), (1.0, -0.1), (1.0, np.nan), (1.0, np.inf),
    (1e308, 1e308),
])
def test_invalid_or_overflowing_distance_time_inputs_are_rejected(distance_config, distance, dt):
    with pytest.raises(ValueError):
        absolute_distance_components(distance, dt, distance_config["reward"])


@pytest.mark.parametrize("coefficient", [0.0, -0.01, np.nan, np.inf, -np.inf])
def test_nonpositive_or_nonfinite_yaml_distance_coefficients_are_rejected(tmp_path, distance_config, coefficient):
    config = deepcopy(distance_config)
    config["reward"]["distance_cost_coefficient"] = coefficient
    with pytest.raises(ValueError):
        _environment(tmp_path, config)


@pytest.mark.parametrize("name,value", [
    ("survival_penalty_per_second", 0.2),
    ("survival_penalty_per_second", np.nan),
    ("collision_penalty", np.inf),
])
def test_nonfinite_or_wrong_sign_shared_coefficients_are_rejected(tmp_path, distance_config, name, value):
    config = deepcopy(distance_config)
    config["reward"][name] = value
    with pytest.raises(ValueError):
        _environment(tmp_path, config)


@pytest.mark.parametrize("name,value", [
    ("active_progress_coefficient", 1.0),
    ("progress_coefficient", 1.0),
    ("time_penalty", -0.01),
])
def test_distance_schema_cannot_mix_with_progress_rewards(tmp_path, distance_config, name, value):
    config = deepcopy(distance_config)
    config["reward"][name] = value
    with pytest.raises(ValueError):
        _environment(tmp_path, config)


@pytest.mark.parametrize("missing", ["distance_cost_coefficient", "survival_penalty_per_second"])
def test_incomplete_distance_schema_is_rejected(tmp_path, distance_config, missing):
    config = deepcopy(distance_config)
    del config["reward"][missing]
    with pytest.raises(ValueError):
        _environment(tmp_path, config)


def test_distance_schema_requires_a_mission(tmp_path, legacy_config, distance_config):
    config = deepcopy(legacy_config)
    config["reward"] = deepcopy(distance_config["reward"])
    with pytest.raises(ValueError):
        _environment(tmp_path, config)


def test_full_random_episode_preserves_states_observations_and_termination(tmp_path, distance_config, baseline_config):
    distance_env = _environment(tmp_path, distance_config)
    baseline_env = _environment(tmp_path, baseline_config, "unchanged_active")
    observations, _ = distance_env.reset(seed=7)
    expected_observations, _ = baseline_env.reset(seed=7)
    np.testing.assert_array_equal(observations, expected_observations)
    total = 0.0
    for _ in range(distance_env.max_episode_steps):
        actions = distance_env.sample_actions()
        actual = distance_env.step(actions)
        expected = baseline_env.step(actions)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(distance_env.get_state(), baseline_env.get_state())
        np.testing.assert_array_equal(distance_env.accelerations, baseline_env.accelerations)
        assert actual[2:4] == expected[2:4]
        assert np.isfinite(actual[0]).all() and np.isfinite(actual[1]).all()
        assert np.isfinite(list(actual[4]["reward_components"].values())).all()
        total += actual[1][0]
        if actual[2] or actual[3]:
            break
    assert actual[2] or actual[3]
    assert distance_env.step_count == baseline_env.step_count
    assert actual[4]["captured"] == expected[4]["captured"]
    assert actual[4]["enemy_attack_success"] == expected[4]["enemy_attack_success"]
    assert actual[1][0] != expected[1][0]
    assert total == pytest.approx(sum(actual[4]["episode_reward_components"].values()))
