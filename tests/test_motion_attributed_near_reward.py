"""Production contracts for motion-attributed record-distance shaping."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.interception_env import InterceptionEnv, REWARD_FIELDS
from envs.near_shaping import motion_attributed_components
from utils.active_near_replay import active_near_shaping


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def legacy_config():
    return yaml.safe_load((ROOT / "configs/interception_1v1_line_cost_k0p1.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def active_config(legacy_config):
    config = deepcopy(legacy_config)
    config["reward"]["best_so_far_shaping"]["mode"] = "motion_attributed"
    config["reward"]["line_cost_coefficient"] = 0.01
    return config


def controlled_environment(config, own_velocity=0.0, attacker_velocity=-20.0):
    env = InterceptionEnv(config, seed=1001)
    env.reset(angle_degrees=0.0)
    env.states[:, :3] = [[100.0, 0.0, 50.0], [120.0, 0.0, 50.0]]
    env.states[:, 3:] = [[own_velocity, 0.0, 0.0], [attacker_velocity, 0.0, 0.0]]
    env.accelerations[:] = 0.0
    env._episode_min_distance = 20.0
    env.attacker_policy = lambda state: np.zeros(3)
    return env


def test_reset_initializes_diagnostics_without_adding_reward_or_observation_fields(active_config):
    env = InterceptionEnv(active_config, seed=1001)
    obs, info = env.reset(angle_degrees=7.0)
    distance = np.linalg.norm(info["states"][1, :3] - info["states"][0, :3])
    assert obs.shape == (16,)
    assert env.motion_attributed_near_enabled
    assert env.reward_fields == REWARD_FIELDS + ("near_shaping", "line_cost")
    assert info["near_counterfactual_distance"] == distance
    assert info["near_shaping_uncorrected"] == info["near_shaping_removed"] == 0.0
    assert info["episode_near_shaping_uncorrected"] == info["episode_near_shaping_removed"] == 0.0
    assert env.get_state()["motion_near_diagnostics"]["counterfactual_distance"] == distance
    _, _, _, _, updated = env.step(np.zeros(3))
    env.reset(angle_degrees=-7.0)
    assert env.get_state()["episode_motion_near_diagnostics"] == dict(uncorrected=0.0, removed=0.0)
    assert updated["near_shaping_uncorrected"] > 0.0


@pytest.mark.parametrize("own_velocity", [0.0, 10.0, -10.0])
@pytest.mark.parametrize("beta", [100.0, 200.0])
def test_actual_motion_matches_one_step_counterfactual(active_config, own_velocity, beta):
    active_config["reward"]["best_so_far_shaping"]["beta"] = beta
    env = controlled_environment(active_config, own_velocity=own_velocity)
    old_position = env.states[0, :3].copy()
    _, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert not (terminated or truncated)
    actual_distance = np.linalg.norm(env.states[1, :3] - env.states[0, :3])
    cf_distance = np.linalg.norm(env.states[1, :3] - old_position)
    assert cf_distance == 18.0
    expected_old = beta * (np.exp(-actual_distance / 20.0) - np.exp(-1.0))
    expected_active = beta * max(0, np.exp(-actual_distance / 20.0) - np.exp(-0.9))
    assert info["near_shaping_uncorrected"] == pytest.approx(expected_old)
    assert info["near_shaping_removed"] == pytest.approx(expected_old - expected_active)
    assert info["reward_components"]["near_shaping"] == pytest.approx(expected_active)
    assert info["episode_near_shaping_uncorrected"] == info["near_shaping_uncorrected"]
    assert info["episode_near_shaping_removed"] == info["near_shaping_removed"]
    assert reward == sum(info["reward_components"].values())
    if own_velocity <= 0:
        assert info["reward_components"]["near_shaping"] == 0
    else:
        assert info["reward_components"]["near_shaping"] > 0


def test_full_stationary_episode_receives_zero_active_near(active_config):
    env = InterceptionEnv(active_config, seed=1001)
    env.reset(angle_degrees=-11.0)
    done = False
    while not done:
        _, _, done, truncated, info = env.step(np.zeros(3))
        assert not truncated
        assert info["reward_components"]["near_shaping"] == 0.0
    assert info["enemy_attack_success"]
    assert info["episode_near_shaping_uncorrected"] > 0
    assert info["episode_near_shaping_removed"] == info["episode_near_shaping_uncorrected"]


def test_equal_and_revisited_record_distances_are_not_reissued(active_config):
    env = controlled_environment(active_config, own_velocity=10.0, attacker_velocity=0.0)
    rewards = []
    for velocity in [10.0] * 5 + [-10.0] * 5 + [10.0] * 6:
        env.states[0, 3] = velocity
        rewards.append(env.step(np.zeros(3))[-1]["reward_components"]["near_shaping"])
    assert all(value > 0 for value in rewards[:5])
    np.testing.assert_array_equal(rewards[5:15], np.zeros(10))
    assert rewards[15] > 0
    assert env.best_distance_so_far == 14.0
    assert sum(rewards) == pytest.approx(100 * (np.exp(-14.0 / 20.0) - np.exp(-1.0)))


@pytest.mark.parametrize("beta", [100.0, 200.0])
def test_random_physical_rollout_matches_shared_offline_formula(active_config, beta):
    active_config["reward"]["best_so_far_shaping"]["beta"] = beta
    env = InterceptionEnv(active_config, seed=1001)
    env.reset(angle_degrees=18.0)
    generator = np.random.default_rng(37)
    states = [env.states.copy()]
    active, original, removed, counterfactual = [], [], [], []
    done = False
    while not done:
        previous_best = env.best_distance_so_far
        previous_pursuer_position = env.states[0, :3].copy()
        observation, reward, done, truncated, info = env.step(generator.uniform(-1, 1, 3))
        assert not truncated
        assert np.isfinite(observation).all() and np.isfinite(reward)
        states.append(env.states.copy())
        active.append(info["reward_components"]["near_shaping"])
        original.append(info["near_shaping_uncorrected"])
        removed.append(info["near_shaping_removed"])
        counterfactual.append(info["near_counterfactual_distance"])
        scalar_cf = float(np.linalg.norm(env.states[1, :3] - previous_pursuer_position))
        scalar = motion_attributed_components(previous_best, env.best_distance_so_far, scalar_cf, 20.0, beta)
        assert active[-1] == float(scalar["active_near"])
        assert original[-1] == float(scalar["original_near"])
        assert removed[-1] == float(scalar["removed_near"])
        assert counterfactual[-1] == scalar_cf
    expected = active_near_shaping(np.asarray(states), 20.0, beta)
    for actual, field in ((active, "active_near"), (original, "original_near"),
                          (removed, "removed_near"), (counterfactual, "counterfactual_distances")):
        # The axis-wise offline norm and scalar online norm can round differently.
        np.testing.assert_allclose(actual, expected[field], rtol=0, atol=1e-12)
    assert info["episode_near_shaping_uncorrected"] == pytest.approx(sum(original))
    assert info["episode_near_shaping_removed"] == pytest.approx(sum(removed))
    assert sum(active) <= sum(original)


def test_legacy_and_explicit_record_mode_preserve_original_arithmetic_and_schema(legacy_config):
    explicit = deepcopy(legacy_config)
    explicit["reward"]["best_so_far_shaping"]["mode"] = "record_minimum"
    environments = [InterceptionEnv(config, seed=1001) for config in (legacy_config, explicit)]
    for env in environments:
        env.reset(angle_degrees=9.0)
        assert not env.motion_attributed_near_enabled
        assert "motion_near_diagnostics" not in env.get_state()
        assert "near_shaping_uncorrected" not in env._info()
    generator = np.random.default_rng(45)
    for _ in range(180):
        action = generator.uniform(-1, 1, 3)
        previous_best = environments[0].best_distance_so_far
        results = [env.step(action) for env in environments]
        np.testing.assert_equal(results[0], results[1])
        expected = float(100 * (np.exp(-environments[0].best_distance_so_far / 20.0) - np.exp(-previous_best / 20.0)))
        assert results[0][-1]["reward_components"]["near_shaping"] == expected


@pytest.mark.parametrize("geometry", ["ongoing", "capture", "failure", "capture_priority"])
def test_reward_mode_preserves_dynamics_observation_and_terminal_rules(legacy_config, active_config, geometry):
    # Align the line coefficient to isolate only the near attribution rule.
    legacy_config["reward"]["line_cost_coefficient"] = active_config["reward"]["line_cost_coefficient"]
    environments = [controlled_environment(config, own_velocity=10.0) for config in (legacy_config, active_config)]
    placements = {
        "ongoing": ([100, 0, 50], [120, 0, 50]),
        "capture": ([100, 0, 50], [106, 0, 50]),
        "failure": ([100, 0, 50], [8, 0, 0]),
        "capture_priority": ([4, 0, 0], [8, 0, 0]),
    }
    for env in environments:
        env.states[:, :3] = placements[geometry]
        env.states[:, 3:] = 0
        if geometry == "capture":
            env.states[0, 3] = 20.0
        env._episode_min_distance = env._distance()
    legacy, active = [env.step(np.zeros(3)) for env in environments]
    np.testing.assert_array_equal(legacy[0], active[0])
    np.testing.assert_array_equal(environments[0].states, environments[1].states)
    assert legacy[2:4] == active[2:4]
    for field in REWARD_FIELDS + ("line_cost",):
        assert legacy[-1]["reward_components"][field] == active[-1]["reward_components"][field]
    assert active[-1]["captured"] == (geometry in ("capture", "capture_priority"))
    assert active[-1]["enemy_attack_success"] == (geometry == "failure")


@pytest.mark.parametrize("beta", [0.0, 100.0, 200.0])
def test_mid_episode_restore_preserves_physics_rng_and_near_accounting(active_config, beta):
    active_config["reward"]["best_so_far_shaping"]["beta"] = beta
    env = InterceptionEnv(active_config, seed=1001)
    env.reset(angle_degrees=-14.0)
    generator = np.random.default_rng(14)
    for _ in range(100):
        env.step(generator.uniform(-1, 1, 3))
    snapshot = env.get_state()
    clone = InterceptionEnv(active_config, seed=999)
    clone.set_state(snapshot)
    np.testing.assert_equal(clone.get_state(), snapshot)
    for _ in range(50):
        action = generator.uniform(-1, 1, 3)
        np.testing.assert_equal(env.step(action), clone.step(action))
        np.testing.assert_equal(env.get_state(), clone.get_state())
    np.testing.assert_equal(env.reset(), clone.reset())


@pytest.mark.parametrize("corruption", ["missing_current", "missing_episode", "negative", "nan", "text", "bool", "complex",
                                        "current_accounting", "episode_accounting", "over_budget", "wrong_fields"])
def test_invalid_motion_snapshot_is_rejected_atomically(active_config, corruption):
    env = controlled_environment(active_config, own_velocity=10.0)
    env.step(np.zeros(3))
    before = env.get_state()
    invalid = deepcopy(before)
    if corruption == "missing_current":
        invalid.pop("motion_near_diagnostics")
    elif corruption == "missing_episode":
        invalid.pop("episode_motion_near_diagnostics")
    elif corruption in ("negative", "nan", "text", "bool", "complex"):
        invalid["motion_near_diagnostics"]["counterfactual_distance"] = {
            "negative": -1.0, "nan": np.nan, "text": "1", "bool": True, "complex": 1j}[corruption]
    elif corruption == "current_accounting":
        invalid["motion_near_diagnostics"]["removed"] = 0.0
    elif corruption == "episode_accounting":
        invalid["episode_motion_near_diagnostics"]["uncorrected"] += 1.0
    elif corruption == "over_budget":
        invalid["episode_motion_near_diagnostics"]["uncorrected"] = 101.0
    else:
        invalid["motion_near_diagnostics"]["extra"] = 1.0
    with pytest.raises(ValueError, match="motion near"):
        env.set_state(invalid)
    np.testing.assert_equal(env.get_state(), before)


@pytest.mark.parametrize("mode", [None, "", "unknown", True, [], {}])
def test_unknown_near_mode_is_rejected(legacy_config, mode):
    legacy_config["reward"]["best_so_far_shaping"]["mode"] = mode
    with pytest.raises(ValueError, match="best_so_far_shaping.mode"):
        InterceptionEnv(legacy_config)


@pytest.mark.parametrize("arguments", [
    (20.0, 21.0, 18.0, 20.0, 100.0), (-1.0, 0.0, 18.0, 20.0, 100.0),
    (20.0, 18.0, np.nan, 20.0, 100.0), (20.0, 18.0, -1.0, 20.0, 100.0),
    ([20.0, 19.0], [18.0, 17.0, 16.0], 18.0, 20.0, 100.0),
    (20.0, 18.0, 18.0, 0.0, 100.0), (20.0, 18.0, 18.0, 20.0, -1.0),
    (20.0, 18.0, 18.0, 20.0, 1j),
])
def test_shared_components_reject_invalid_values(arguments):
    with pytest.raises(ValueError):
        motion_attributed_components(*arguments)
