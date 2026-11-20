"""1v1 physical/observation/terminal acceptance with explicit test-only scales."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.dynamics import integrate, scale_action
from envs.interception_env import InterceptionEnv, OBSERVATION_FIELDS, REWARD_FIELDS


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    result = yaml.safe_load((ROOT / "configs/baseline_capture500.yaml").read_text(encoding="utf-8"))
    result["environment"]["num_pursuers"] = 1
    initialization = result["initialization"]
    initialization["pursuer_position"] = initialization.pop("pursuer_positions")[0]
    initialization["pursuer_velocity"] = initialization.pop("pursuer_velocities")[0]
    initialization["angle_range_degrees"] = [-30.0, 30.0]
    result["reward"].pop("collision_penalty")
    # An explicit unit-test fixture exercises the division contract. These
    # divisors do not declare the user's pending formal scale table confirmed.
    result["normalization"] = dict(confirmed=True, acceleration_kind="applied",
        scales=[20., 20., 8., 3., 3., 2., 1500., 1500., 1500., 40., 40., 16., 750., 750., 750., 1.])
    return result


def equal(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            equal(left, right)
    else:
        assert actual == expected


def stationary_attacker(env):
    # Terminal geometry fixtures isolate strict inequalities from target motion.
    env.attacker_policy = lambda state: np.zeros(3)


def test_unconfirmed_scale_table_is_a_formal_startup_blocker(config):
    config["normalization"] = dict(confirmed=False, acceleration_kind=None, scales=None)
    with pytest.raises(ValueError, match="explicitly confirmed"):
        InterceptionEnv(config)


@pytest.mark.parametrize("scales", [None, [1.] * 15, [1.] * 17, [0.] * 16,
                                    [-1.] * 16, [np.nan] * 16, [np.inf] * 16])
def test_invalid_or_missing_scale_vectors_are_rejected(config, scales):
    config["normalization"]["scales"] = scales
    with pytest.raises(ValueError):
        InterceptionEnv(config)


@pytest.mark.parametrize("kind", [None, "command", "unknown", False])
def test_acceleration_semantics_are_explicit(config, kind):
    config["normalization"]["acceleration_kind"] = kind
    with pytest.raises(ValueError, match="acceleration_kind"):
        InterceptionEnv(config)


def test_reset_has_exact_p0_zero_motion_and_one_normalization_count(config):
    env = InterceptionEnv(config, seed=1001)
    obs, info = env.reset(angle_degrees=0)
    assert obs.shape == (16,) and obs.dtype == np.float32 and np.isfinite(obs).all()
    assert env.states.shape == (2, 6) and env.action_shape == (3,)
    np.testing.assert_array_equal(env.states[0, :3], [1.5, 0, 1])
    np.testing.assert_array_equal(env.states[1, :3], [497.49371855331, 0, 50])
    np.testing.assert_array_equal(env.states[:, 3:], np.zeros((2, 3)))
    np.testing.assert_array_equal(env.accelerations, np.zeros((2, 3)))
    assert len(OBSERVATION_FIELDS) == 16 and info["normalization_run_observations"] == 1
    assert info["normalization_episode_observations"] == 1 and info["raw_observation"][-1] == 1
    assert info["angle_degrees"] == info["initial_angle_degrees"] == 0


def test_exact_field_order_and_world_frame(config):
    env = InterceptionEnv(config)
    env.reset(angle_degrees=0)
    env.states = np.array([[11., 12., 13., 1., 2., 3.], [21., 32., 43., -1., -2., -3.]])
    env.accelerations[0] = [.2, .3, .4]
    env.step_count = 125
    raw = env.raw_observation()
    expected = np.array([1, 2, 3, .2, .3, .4, 10, 20, 30, -2, -4, -6, 21, 32, 43, .75])
    np.testing.assert_array_equal(raw, expected)


def test_angles_use_independent_continuous_uniform_draws(config):
    env = InterceptionEnv(config, seed=1001)
    expected_rng = np.random.default_rng(1001)
    values = []
    for _ in range(100):
        env.reset()
        expected = expected_rng.uniform(-30, 30)
        assert env.angle_degrees == expected
        theta = np.deg2rad(expected)
        position = env.states[1, :3]
        np.testing.assert_allclose(position, [497.49371855331 * np.cos(theta),
                                             497.49371855331 * np.sin(theta), 50])
        assert np.isclose(np.linalg.norm(position), 500)
        values.append(expected)
    assert len(set(values)) == 100
    assert any(value > 0 for value in values) and any(value < 0 for value in values)


def test_explicit_positive_angle_is_counterclockwise_and_does_not_consume_rng(config):
    env = InterceptionEnv(config, seed=19)
    env.reset(angle_degrees=30)
    assert env.states[1, 0] > 0 and env.states[1, 1] > 0 and env.states[1, 2] == 50
    env.reset()
    assert env.angle_degrees == np.random.default_rng(19).uniform(-30, 30)


def test_environment_seed_and_state_do_not_alias_or_change_global_numpy_rng(config):
    global_before = deepcopy(np.random.get_state())
    left = InterceptionEnv(config, seed=7)
    right = InterceptionEnv(config, seed=8)
    left.reset()
    right.reset()
    right_snapshot = right.get_state()
    left.step(np.ones(3))
    equal(right.get_state(), right_snapshot)
    equal(np.random.get_state(), global_before)
    assert left.angle_degrees != right.angle_degrees


def test_dynamics_match_existing_integrator_and_after_jerk_observation(config):
    env = InterceptionEnv(config)
    env.reset(angle_degrees=0)
    old = env.states.copy()
    action = np.array([1., 0., 0.])
    expected_p, applied_p = integrate(old[0], scale_action(action, env.pursuer_limits),
                                      np.zeros(3), env.dt, env.pursuer_limits)
    expected_a, applied_a = integrate(old[1], env.attacker_policy(old[1]),
                                      np.zeros(3), env.dt, env.attacker_limits)
    obs, reward, terminated, truncated, info = env.step(action)
    np.testing.assert_array_equal(env.states, np.vstack((expected_p, expected_a)))
    np.testing.assert_array_equal(env.accelerations, np.vstack((applied_p, applied_a)))
    np.testing.assert_array_equal(info["raw_observation"][3:6], applied_p)
    np.testing.assert_allclose(applied_p, [.5, 0, 0])
    np.testing.assert_array_equal(obs, (info["raw_observation"] / env.scales).astype(np.float32))
    assert np.isfinite(reward) and not terminated and not truncated


def test_applied_and_realized_acceleration_diverge_at_speed_limit(config):
    for kind in ("applied", "realized"):
        chosen = deepcopy(config)
        chosen["normalization"]["acceleration_kind"] = kind
        env = InterceptionEnv(chosen)
        env.reset(angle_degrees=0)
        env.states[0, 3:] = [20, 0, 0]
        _, _, _, _, info = env.step([1, 0, 0])
        np.testing.assert_allclose(info["applied_accelerations"][0], [.5, 0, 0])
        np.testing.assert_array_equal(info["realized_accelerations"][0], [0, 0, 0])
        expected = info[kind + "_accelerations"][0]
        np.testing.assert_array_equal(info["raw_observation"][3:6], expected)


def test_random_policy_obeys_existing_physical_constraints(config):
    env = InterceptionEnv(config, seed=1001)
    env.reset()
    action_rng = np.random.default_rng(42)
    for _ in range(200):
        before = env.accelerations.copy()
        obs, reward, done, truncated, _ = env.step(action_rng.uniform(-1, 1, size=3))
        assert np.isfinite(obs).all() and np.isfinite(env.states).all() and np.isfinite(reward)
        assert not truncated
        for i, limits in enumerate((env.pursuer_limits, env.attacker_limits)):
            assert np.linalg.norm(env.states[i, 3:5]) <= limits.max_horizontal_speed + 1e-10
            assert -limits.max_descent_speed <= env.states[i, 5] <= limits.max_climb_speed
            assert np.linalg.norm(env.accelerations[i, :2]) <= limits.max_horizontal_acceleration + 1e-10
            assert abs(env.accelerations[i, 2]) <= limits.max_vertical_acceleration + 1e-10
            assert np.linalg.norm(env.accelerations[i] - before[i]) <= limits.max_jerk * env.dt + 1e-10
        if done:
            break


@pytest.mark.parametrize("distance,captured", [(4.999999, True), (5., False), (5.000001, False)])
def test_capture_uses_strict_five_metre_distance(config, distance, captured):
    env = InterceptionEnv(config)
    env.reset(angle_degrees=0)
    stationary_attacker(env)
    env.states[:, :3] = [[100 - distance, 0, 0], [100, 0, 0]]
    _, _, done, _, info = env.step(np.zeros(3))
    assert info["captured"] is captured and done is captured


@pytest.mark.parametrize("distance,failed", [(9.999999, True), (10., False), (10.000001, False)])
def test_defense_failure_uses_strict_ten_metre_distance(config, distance, failed):
    env = InterceptionEnv(config)
    env.reset(angle_degrees=0)
    stationary_attacker(env)
    env.states[:, :3] = [[100, 0, 0], [distance, 0, 0]]
    _, _, done, _, info = env.step(np.zeros(3))
    assert info["enemy_attack_success"] is failed and done is failed


def test_capture_wins_simultaneous_failure_and_timeout(config):
    config["environment"]["max_episode_steps"] = 1
    env = InterceptionEnv(config)
    env.reset(angle_degrees=0)
    stationary_attacker(env)
    env.states[:, :3] = [[0, 0, 0], [1, 0, 0]]
    _, reward, done, truncated, info = env.step(np.zeros(3))
    assert done and not truncated and info["captured"] and not info["enemy_attack_success"]
    assert not info["timeout"] and info["capture_time"] == .1
    assert set(info["reward_components"]) == set(REWARD_FIELDS)
    assert np.isclose(reward, 500 - .01 * .1 - .2 * .1)
    assert reward == sum(info["reward_components"].values())
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3))


def test_fifty_second_timeout_is_true_terminal_and_rollout_cut_is_not(config):
    env = InterceptionEnv(config)
    env.reset(angle_degrees=0)
    stationary_attacker(env)
    for step in range(1, 501):
        obs, _, done, truncated, info = env.step(np.zeros(3))
        assert not truncated
        assert done is (step == 500)
        if step == 512 // 8:
            assert not info["terminated"]
    assert info["timeout"] and info["outcome"] == "Timeout" and info["terminated"]
    assert not info["captured"] and not info["enemy_attack_success"] and info["capture_time"] is None
    assert obs[-1] == 0 and env.step_count * env.dt == 50


def test_normalization_never_clips_and_queries_do_not_increment_diagnostics(config):
    config["normalization"]["scales"] = [1.] * 16
    env = InterceptionEnv(config)
    obs, info = env.reset(angle_degrees=0)
    assert obs[6] > 400
    expected_fields = [6, 8, 12, 14]
    assert info["normalization_violations"]["field_indices"] == expected_fields
    np.testing.assert_array_equal(np.flatnonzero(info["normalization_violation_mask"]), expected_fields)
    snapshot = env.get_state()
    for _ in range(3):
        env.raw_observation()
        env._info()
        equal(env.get_state(), snapshot)
    env.step(np.zeros(3))
    _, after_reset = env.reset(angle_degrees=0)
    assert after_reset["normalization_run_observations"] == 3
    assert after_reset["normalization_episode_observations"] == 1
    assert after_reset["normalization_run_violation_counts"][6] == 3
    assert after_reset["normalization_episode_violation_counts"][6] == 1


def test_boundary_clip_retains_velocity_and_records_actual_clip_events(config):
    env = InterceptionEnv(config)
    env.reset(angle_degrees=0)
    env.states[0, :3] = [749.99, 0, 1]
    env.states[0, 3:] = [20, 0, 0]
    _, _, _, _, info = env.step(np.zeros(3))
    assert env.states[0, 0] == 750 and env.states[0, 3] == 20
    assert info["boundary_clip_count"] == 1
    _, _, _, _, info = env.step(np.zeros(3))
    assert info["boundary_clip_counts"][0] == 2


def test_snapshot_restores_partial_episode_rng_accelerations_and_diagnostics(config):
    env = InterceptionEnv(config, seed=1001)
    env.reset()
    actions = np.random.default_rng(5).uniform(-1, 1, (40, 3))
    for action in actions[:13]:
        env.step(action)
    saved = env.get_state()
    clone = InterceptionEnv(config, seed=999)
    clone.set_state(saved)
    equal(env.get_state(), clone.get_state())
    for action in actions[13:]:
        expected = env.step(action)
        actual = clone.step(action)
        equal(actual, expected)
    equal(env.reset(), clone.reset())
    equal(env.get_state(), clone.get_state())
    saved["states"][:] = 0
    assert np.any(env.states != 0)


def test_invalid_restore_is_atomic_and_configuration_is_protected(config):
    env = InterceptionEnv(config, seed=10)
    env.reset()
    before = env.get_state()
    corrupted = deepcopy(before)
    corrupted["states"][0, 0] = np.nan
    with pytest.raises(ValueError):
        env.set_state(corrupted)
    equal(env.get_state(), before)
    other_config = deepcopy(config)
    other_config["capture"]["radius"] = 4
    other = InterceptionEnv(other_config)
    with pytest.raises(ValueError, match="configuration"):
        other.set_state(before)


def test_config_and_returned_info_are_independent_copies(config):
    env = InterceptionEnv(config)
    _, info = env.reset(angle_degrees=0)
    config["normalization"]["scales"][0] = 123
    info["states"][:] = 0
    info["raw_observation"][:] = 0
    info["normalization_run_violation_counts"][:] = 999
    assert env.scales[0] == 20 and np.any(env.states != 0)
    assert not np.all(env._run_violation_counts == 999)


@pytest.mark.parametrize("action", [[0, 0], [0, 0, 0, 0], [0, np.nan, 0], [np.inf, 0, 0]])
def test_invalid_actions_are_rejected(config, action):
    env = InterceptionEnv(config)
    env.reset(angle_degrees=0)
    with pytest.raises(ValueError):
        env.step(action)
