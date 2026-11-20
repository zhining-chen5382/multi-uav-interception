"""Reward attribution, time units, accounting and historical compatibility."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from envs import PursuitEnv
from envs.rewards import active_progress_components
from utils.evaluation import run_episode


WEIGHTED_COMPONENTS = (
    "active_progress_reward", "survival", "collision", "capture", "attack_failure",
)
RAW_COMPONENTS = (
    "pursuer_attacker_progress", "attacker_base_progress", "active_progress",
)


@pytest.mark.parametrize("current_distance,expected_pe,expected_active", [
    (98.0, 2.0, 0.0),
    (95.0, 5.0, 3.0),
    (99.0, 1.0, -1.0),
], ids=["attacker_only", "additional_pursuer_progress", "less_than_natural_advance"])
def test_active_progress_removes_attacker_base_advance(
    baseline_config, current_distance, expected_pe, expected_active,
):
    parts = active_progress_components(
        previous_min_distance=100.0, min_distance=current_distance,
        previous_base_distance=200.0, base_distance=198.0,
        dt=baseline_config["environment"]["dt"], coefficients=baseline_config["reward"],
    )
    assert parts["pursuer_attacker_progress"] == expected_pe
    assert parts["attacker_base_progress"] == 2.0
    assert parts["active_progress"] == expected_active
    assert parts["active_progress_reward"] == expected_active
    assert set(parts) == set(RAW_COMPONENTS + WEIGHTED_COMPONENTS)
    assert np.isfinite(list(parts.values())).all()


@pytest.mark.parametrize("dt,expected", [(0.1, -0.02), (0.25, -0.05), (0.5, -0.1)])
def test_survival_penalty_is_per_physical_second(baseline_config, dt, expected):
    parts = active_progress_components(100.0, 100.0, 200.0, 200.0,
                                       dt, baseline_config["reward"])
    assert parts["survival"] == pytest.approx(expected)
    assert parts["active_progress_reward"] == 0.0
    assert parts["capture"] == parts["attack_failure"] == 0.0


def test_reward_coefficients_scale_progress_and_survival_from_configuration(baseline_config):
    coefficients = deepcopy(baseline_config["reward"])
    coefficients.update(active_progress_coefficient=2.5,
                        survival_penalty_per_second=-0.7)
    parts = active_progress_components(100.0, 95.0, 200.0, 198.0,
                                       0.2, coefficients)
    assert parts["active_progress"] == 3.0
    assert parts["active_progress_reward"] == 7.5
    assert parts["survival"] == pytest.approx(-0.14)


@pytest.mark.parametrize("captured,attacked,expected_capture,expected_failure", [
    (False, True, 0.0, -100.0),
    (True, False, 100.0, 0.0),
])
def test_terminal_rewards_keep_original_scale(
    baseline_config, captured, attacked, expected_capture, expected_failure,
):
    parts = active_progress_components(100.0, 98.0, 200.0, 198.0,
                                       0.1, baseline_config["reward"],
                                       captured=captured, attacker_reached_base=attacked)
    assert parts["capture"] == expected_capture
    assert parts["attack_failure"] == expected_failure
    assert parts["active_progress_reward"] == 0.0
    expected = expected_capture + expected_failure - 0.02
    assert sum(parts[key] for key in WEIGHTED_COMPONENTS) == pytest.approx(expected)


@pytest.mark.parametrize("unsafe_pairs", [1, 3])
def test_collision_remains_one_team_penalty_per_step(baseline_config, unsafe_pairs):
    parts = active_progress_components(100.0, 98.0, 200.0, 198.0,
                                       0.1, baseline_config["reward"],
                                       collision_pairs=unsafe_pairs)
    assert parts["collision"] == -5.0


def test_environment_reward_counts_only_weighted_components_and_keeps_inputs():
    env = PursuitEnv()
    observations, previous = env.reset(seed=5)
    assert observations.shape == (3, 21)
    assert previous["global_state"].shape == env.get_state().shape == (24,)
    for _ in range(8):
        observations, rewards, terminated, truncated, info = env.step(
            np.array([[1.0, 0.2, 0.0]] * 3)
        )
        parts = info["reward_components"]
        expected_pe = previous["min_distance"] - info["min_distance"]
        expected_eb = previous["distance_to_base"] - info["distance_to_base"]
        assert parts["pursuer_attacker_progress"] == pytest.approx(expected_pe)
        assert parts["attacker_base_progress"] == pytest.approx(expected_eb)
        assert parts["active_progress"] == pytest.approx(expected_pe - expected_eb)
        assert parts["active_progress_reward"] == pytest.approx(
            env.config["reward"]["active_progress_coefficient"] * (expected_pe - expected_eb)
        )
        assert parts["survival"] == pytest.approx(-0.02)
        assert "progress" not in parts and "time" not in parts
        expected_reward = sum(parts[name] for name in WEIGHTED_COMPONENTS)
        np.testing.assert_allclose(rewards, [expected_reward] * 3)
        assert observations.shape == (3, 21)
        assert env.get_state().shape == info["global_state"].shape == (24,)
        assert np.isfinite(observations).all() and np.isfinite(rewards).all()
        assert not terminated and not truncated
        previous = info
    # Raw distance diagnostics must never be added into the team return.
    assert not np.isclose(rewards[0], sum(parts.values()))
    assert "progress_coefficient" not in env.config["reward"]
    assert "time_penalty" not in env.config["reward"]


def test_environment_uses_yaml_progress_and_physical_time_parameters(config_factory):
    env = PursuitEnv(config_factory(
        environment={"dt": 0.2},
        reward={"active_progress_coefficient": 2.5,
                "survival_penalty_per_second": -0.7},
    ))
    _, before = env.reset(seed=0)
    _, reward, _, _, after = env.step(np.array([[1.0, 0.0, 0.0]] * 3))
    expected_progress = 2.5 * (
        before["min_distance"] - after["min_distance"]
        - (before["distance_to_base"] - after["distance_to_base"])
    )
    assert after["reward_components"]["active_progress_reward"] == pytest.approx(expected_progress)
    assert after["reward_components"]["survival"] == pytest.approx(-0.14)
    np.testing.assert_allclose(reward, [expected_progress - 0.14] * 3)


def test_episode_component_totals_are_copies_and_reset_between_episodes():
    env = PursuitEnv()
    _, initial = env.reset(seed=0)
    expected = dict.fromkeys(WEIGHTED_COMPONENTS, 0.0)
    assert initial["episode_reward_components"] == expected
    reward_sum = 0.0
    for step in range(5):
        _, reward, _, _, info = env.step(env.sample_actions())
        reward_sum += reward[0]
        for name in expected:
            expected[name] += info["reward_components"][name]
        assert info["episode_reward_components"] == pytest.approx(expected)
        if step == 0:
            info["episode_reward_components"]["survival"] = -10000.0
    assert sum(expected.values()) == pytest.approx(reward_sum)
    assert initial["episode_reward_components"] == dict.fromkeys(WEIGHTED_COMPONENTS, 0.0)
    _, reset = env.reset(seed=0)
    assert reset["episode_reward_components"] == dict.fromkeys(WEIGHTED_COMPONENTS, 0.0)


def test_stationary_pursuers_do_not_receive_large_positive_failure_return():
    env = PursuitEnv()
    env.reset(seed=0)
    total = 0.0
    for _ in range(env.max_episode_steps):
        _, reward, terminated, truncated, info = env.step(np.zeros((3, 3)))
        total += reward[0]
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["enemy_attack_success"] and not info["captured"]
    totals = info["episode_reward_components"]
    assert abs(totals["active_progress_reward"]) < env.safety_distance
    assert totals["active_progress_reward"] <= 0.0
    assert totals["survival"] == pytest.approx(-0.2 * env.dt * env.step_count)
    assert totals["capture"] == 0.0 and totals["attack_failure"] == -100.0
    assert totals["collision"] == 0.0
    assert total == pytest.approx(sum(totals.values()))
    assert total < -100.0


@pytest.mark.parametrize("old_key,value", [("progress_coefficient", 1.0), ("time_penalty", -0.01)])
def test_mixed_old_and_active_reward_schemas_are_rejected(config_factory, old_key, value):
    with pytest.raises(ValueError):
        PursuitEnv(config_factory(reward={old_key: value}))


@pytest.mark.parametrize("missing", ["active_progress_coefficient", "survival_penalty_per_second"])
def test_incomplete_active_schema_is_rejected(tmp_path, baseline_config, missing):
    config = deepcopy(baseline_config)
    del config["reward"][missing]
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        PursuitEnv(path)


@pytest.mark.parametrize("name,value", [
    ("active_progress_coefficient", -1.0),
    ("active_progress_coefficient", np.nan),
    ("active_progress_coefficient", np.inf),
    ("survival_penalty_per_second", 0.2),
    ("survival_penalty_per_second", np.nan),
    ("survival_penalty_per_second", np.inf),
])
def test_invalid_active_reward_coefficients_are_rejected(config_factory, name, value):
    with pytest.raises(ValueError):
        PursuitEnv(config_factory(reward={name: value}))


def test_active_progress_requires_a_base(tmp_path, legacy_config, baseline_config):
    config = deepcopy(legacy_config)
    config["reward"] = deepcopy(baseline_config["reward"])
    path = tmp_path / "no_mission.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        PursuitEnv(path)


def test_historical_point_defense_reward_snapshot_is_not_reinterpreted(
    tmp_path, baseline_config,
):
    config = deepcopy(baseline_config)
    config["reward"] = {
        "capture_reward": 100.0, "attack_failure_penalty": -100.0,
        "progress_coefficient": 1.0, "time_penalty": -0.01,
        "collision_penalty": -5.0,
    }
    path = tmp_path / "historical_point_defense.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = PursuitEnv(path)
    _, before = env.reset(seed=0)
    _, reward, _, _, after = env.step(np.zeros((3, 3)))
    expected = before["min_distance"] - after["min_distance"] - 0.01
    np.testing.assert_allclose(reward, [expected] * 3)
    assert set(after["reward_components"]) == {
        "progress", "time", "capture", "attack_failure", "collision",
    }
    assert after["reward_components"]["progress"] > 0.0
    assert after["episode_reward_components"] == pytest.approx(after["reward_components"])


def test_legacy_evaluation_does_not_report_missing_active_terms_as_measured_zero(
    legacy_config_factory,
):
    class ZeroActionAgent:
        def act(self, observations, deterministic=False):
            assert deterministic and observations.shape == (3, 21)
            return np.zeros((3, 3)), None, None, None

    env = PursuitEnv(legacy_config_factory(environment={"max_episode_steps": 1}))
    metrics, trajectory = run_episode(env, ZeroActionAgent(), seed=0)
    assert metrics["cumulative_active_progress_reward"] is None
    assert metrics["cumulative_survival_penalty"] is None
    assert metrics["terminal_reward"] == 0.0
    names = trajectory["reward_component_names"].tolist()
    assert "progress" in names and "time" in names
    assert "active_progress_reward" not in names and "survival" not in names
    np.testing.assert_allclose(trajectory["reward_components"].sum(axis=1), trajectory["rewards"])
