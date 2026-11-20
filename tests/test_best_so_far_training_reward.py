"""Production environment contracts for the controlled best-distance reward."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.interception_env import InterceptionEnv, REWARD_FIELDS
from scripts.offline_best_so_far_reward_replay import best_so_far_shaping


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base_config():
    return yaml.safe_load((ROOT / "configs/interception_1v1.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def shaped_config(base_config):
    result = deepcopy(base_config)
    result["reward"]["best_so_far_shaping"] = dict(length_scale_m=20.0, beta=100.0)
    return result


def stationary_environment(config):
    env = InterceptionEnv(config, seed=1001)
    env.reset(angle_degrees=0.0)
    env.attacker_policy = lambda state: np.zeros(3)
    return env


def place_aircraft(env, pursuer, attacker):
    """Controlled geometry fixture isolates rewards from motion integration."""
    env.states[:, :3] = [pursuer, attacker]
    env.states[:, 3:] = 0
    env.accelerations[:] = 0


def step_at_distance(env, distance):
    place_aircraft(env, [100, 0, 50], [100 + distance, 0, 50])
    return env.step(np.zeros(3))


def test_reset_initializes_formal_history_and_zero_component(shaped_config):
    env = InterceptionEnv(shaped_config, seed=1001)
    with pytest.raises(RuntimeError, match="reset"):
        _ = env.best_distance_so_far
    observation, info = env.reset(angle_degrees=0.0)
    initial = float(np.linalg.norm(env.states[1, :3] - env.states[0, :3]))
    assert observation.shape == (16,)
    assert env.best_distance_so_far == initial == info["best_distance_so_far"]
    assert env.reward_fields == REWARD_FIELDS + ("near_shaping",)
    assert info["reward_components"]["near_shaping"] == 0
    assert info["episode_reward_components"]["near_shaping"] == 0
    assert env.get_state()["best_distance_so_far"] == initial


def test_new_records_match_offline_formula_and_prevent_repeat_rewards(shaped_config):
    env = stationary_environment(shaped_config)
    distances = [env.best_distance_so_far, 150, 25, 8, 20, 12, 8, 7, 7, 30, 6]
    rewards = []
    for distance in distances[1:]:
        _, _, terminated, truncated, info = step_at_distance(env, distance)
        assert not (terminated or truncated)
        rewards.append(info["reward_components"]["near_shaping"])
    expected = best_so_far_shaping(distances, 20, 100)
    np.testing.assert_array_equal(rewards, expected)
    assert rewards[0] > 0 and rewards[1] > 0 and rewards[2] > 0
    np.testing.assert_array_equal(np.asarray(rewards)[3:6], [0, 0, 0])
    assert rewards[6] > 0 and rewards[7] == 0 and rewards[8] == 0 and rewards[9] > 0
    assert env.best_distance_so_far == 6
    assert info["episode_reward_components"]["near_shaping"] == sum(rewards)
    assert sum(rewards) == pytest.approx(100 * (np.exp(-6 / 20) - np.exp(-distances[0] / 20)))


def test_reset_starts_new_shaping_budget(shaped_config):
    env = stationary_environment(shaped_config)
    step_at_distance(env, 8)
    _, info = env.reset(angle_degrees=17.0)
    initial = np.linalg.norm(env.states[1, :3] - env.states[0, :3])
    assert env.best_distance_so_far == initial > 8
    assert info["episode_reward_components"]["near_shaping"] == 0
    _, _, _, _, info = step_at_distance(env, 8)
    assert info["reward_components"]["near_shaping"] > 0


@pytest.mark.parametrize("pursuer,attacker,captured,failed", [
    ([100, 0, 50], [130, 0, 50], False, False),
    ([100, 0, 50], [104, 0, 50], True, False),
    ([100, 0, 50], [8, 0, 0], False, True),
    ([4, 0, 0], [8, 0, 0], True, False),
])
def test_original_reward_observation_dynamics_and_outcomes_unchanged(
        base_config, shaped_config, pursuer, attacker, captured, failed):
    baseline = stationary_environment(base_config)
    shaped = stationary_environment(shaped_config)
    for env in (baseline, shaped):
        place_aircraft(env, pursuer, attacker)
    obs_a, reward_a, done_a, truncated_a, info_a = baseline.step(np.zeros(3))
    obs_b, reward_b, done_b, truncated_b, info_b = shaped.step(np.zeros(3))
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(baseline.states, shaped.states)
    assert done_a == done_b == (captured or failed)
    assert truncated_a == truncated_b == False
    assert info_b["captured"] == captured
    assert info_b["enemy_attack_success"] == failed
    for name in REWARD_FIELDS:
        assert info_b["reward_components"][name] == info_a["reward_components"][name]
    assert info_b["reward_components"]["capture"] == (500 if captured else 0)
    assert info_b["reward_components"]["attack_failure"] == (-100 if failed else 0)
    assert reward_b == reward_a + info_b["reward_components"]["near_shaping"]


def test_actual_random_rollout_exactly_matches_offline_shaping(shaped_config):
    env = InterceptionEnv(shaped_config, seed=1001)
    env.reset(angle_degrees=-12.0)
    generator = np.random.default_rng(938)
    distances = [env.best_distance_so_far]
    rewards = []
    terminated = False
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(generator.uniform(-1, 1, 3))
        assert not truncated
        assert observation.shape == (16,) and np.isfinite(observation).all()
        assert np.isfinite(reward)
        distances.append(info["min_distance"])
        rewards.append(info["reward_components"]["near_shaping"])
    np.testing.assert_array_equal(rewards, best_so_far_shaping(distances, 20, 100))
    assert env.best_distance_so_far == min(distances)


def test_restore_preserves_history_after_flyby_and_later_record(shaped_config):
    env = stationary_environment(shaped_config)
    for distance in (150, 20, 8, 40):
        step_at_distance(env, distance)
    snapshot = env.get_state()
    assert snapshot["best_distance_so_far"] == 8
    clone = stationary_environment(shaped_config)
    clone.set_state(snapshot)
    np.testing.assert_equal(clone.get_state(), snapshot)
    for distance in (30, 8, 7, 50, 6):
        actual = step_at_distance(env, distance)
        restored = step_at_distance(clone, distance)
        np.testing.assert_equal(actual, restored)
        np.testing.assert_equal(env.get_state(), clone.get_state())
    env.reset()
    clone.reset()
    np.testing.assert_equal(env.get_state(), clone.get_state())


@pytest.mark.parametrize("corruption", ["missing", "nan", "infinity", "negative", "bool", "text", "complex", "mismatch", "above_current"])
def test_invalid_history_snapshot_rejected_atomically(shaped_config, corruption):
    env = stationary_environment(shaped_config)
    step_at_distance(env, 20)
    before = env.get_state()
    invalid = deepcopy(before)
    values = dict(nan=np.nan, infinity=np.inf, negative=-1.0, bool=True,
                  text="invalid", complex=1j, mismatch=19.0)
    if corruption == "missing":
        invalid.pop("best_distance_so_far")
    elif corruption == "above_current":
        invalid["best_distance_so_far"] = invalid["episode_min_distance"] = 21.0
    else:
        invalid["best_distance_so_far"] = values[corruption]
    with pytest.raises(ValueError, match="best_distance_so_far"):
        env.set_state(invalid)
    np.testing.assert_equal(env.get_state(), before)


def test_default_reward_and_legacy_snapshot_keep_original_schema(base_config):
    env = InterceptionEnv(base_config, seed=1001)
    env.reset(angle_degrees=-5.0)
    env.step([0.2, -0.1, 0.3])
    snapshot = env.get_state()
    assert env.reward_fields == REWARD_FIELDS
    assert set(snapshot["reward_components"]) == set(REWARD_FIELDS)
    assert "best_distance_so_far" not in snapshot
    clone = InterceptionEnv(base_config, seed=55)
    clone.set_state(snapshot)
    assert clone.best_distance_so_far == snapshot["episode_min_distance"]
    np.testing.assert_equal(clone.get_state(), snapshot)
    np.testing.assert_equal(env.step([0.1, 0.2, -0.3]), clone.step([0.1, 0.2, -0.3]))


@pytest.mark.parametrize("shaping", [None, {}, {"length_scale_m": 20},
    {"length_scale_m": 20, "beta": 100, "extra": 1},
    {"length_scale_m": 0, "beta": 100}, {"length_scale_m": np.inf, "beta": 100},
    {"length_scale_m": None, "beta": 100}, {"length_scale_m": True, "beta": 100},
    {"length_scale_m": 20, "beta": -1}, {"length_scale_m": 20, "beta": np.nan},
    {"length_scale_m": 20, "beta": None}, {"length_scale_m": 20, "beta": False}])
def test_invalid_shaping_configuration_is_rejected(base_config, shaping):
    config = deepcopy(base_config)
    config["reward"]["best_so_far_shaping"] = shaping
    with pytest.raises(ValueError, match="best_so_far_shaping"):
        InterceptionEnv(config)
