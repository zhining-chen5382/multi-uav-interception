"""Engineering checks for the fixed-zero start and seven-stage direction route.

Tiny fixtures consume at most 56 training steps; their injected evaluation
outcomes verify control flow, not real interception performance.
"""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from algorithms.mappo import MAPPOAgent, RolloutBuffer
from envs import PursuitEnv
from utils.direction_curriculum import (
    DirectionCurriculumEnv, evaluation_angles, stage_evaluation_settings,
)
from utils.experiment import set_seed


ROOT = Path(__file__).resolve().parents[1]
ROUTE = [0.0, 2.0, 5.0, 10.0, 30.0, 60.0, 90.0]
MODEL_KEYS = ("actor", "critic", "actor_optimizer", "critic_optimizer")


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _summary(cap=0.0, episodes=1, capture_time=1.0):
    return dict(episodes=episodes, capture_rate=cap, enemy_attack_success_rate=1.0-cap,
                average_capture_time=None if cap == 0 else capture_time,
                average_min_distance=4.0 if cap else 40.0, collision_episode_rate=0.0)


@pytest.fixture
def short_config(tmp_path, monkeypatch):
    from scripts import train_direction_curriculum

    config = _yaml(ROOT / "configs/mappo_direction_curriculum_zero_start.yaml")
    environment = _yaml(ROOT / "configs/baseline_capture500.yaml")
    environment["environment"]["max_episode_steps"] = 3
    env_path = tmp_path / "environment.yaml"
    env_path.write_text(yaml.safe_dump(environment, sort_keys=False), encoding="utf-8")
    config["env_config"] = str(env_path)
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 2
    config["training"]["rollout_steps"] = 3
    config["curriculum"]["max_stage_steps"] = 8
    config["curriculum"]["evaluation_interval_steps"] = 4
    config["curriculum"]["evaluation_episodes"] = 4
    config["training"]["total_env_steps"] = 8 * len(ROUTE)
    monkeypatch.setattr(train_direction_curriculum, "run_post_training_analysis", lambda *args: {"status": "disabled"})
    return config


def _train(tmp_path, config, evaluator):
    from scripts.train_direction_curriculum import train_curriculum

    config = deepcopy(config)
    config["output_dir"] = str(tmp_path / "run")
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = train_curriculum(path, evaluation_fn=evaluator)
    return tmp_path / "run", result


def _stage_number(path):
    part = next(part for part in Path(path).parts if part.startswith("stage_") and "_pm" in part)
    return int(part.split("_")[1])


def test_formal_zero_start_configuration_is_new_and_keeps_v08_learning_settings():
    from scripts.train_direction_curriculum import validate_config

    old = _yaml(ROOT / "configs/mappo_direction_curriculum.yaml")
    config = _yaml(ROOT / "configs/mappo_direction_curriculum_zero_start.yaml")
    v08 = _yaml(ROOT / "configs/mappo_capture500.yaml")
    assert config["curriculum"]["stage_half_angles_degrees"] == ROUTE
    assert config["curriculum"]["max_stage_steps"] == 100000
    assert config["training"]["total_env_steps"] == 700000
    assert config["curriculum"]["evaluation_interval_steps"] == 10000
    assert config["curriculum"]["evaluation_episodes"] == 100
    assert config["curriculum"]["zero_angle_evaluation_episodes"] == 1
    assert config["curriculum"]["zero_angle_promotion_capture_rate"] == 1.0
    assert config["curriculum"]["promotion_capture_rate"] == 0.8
    assert config["curriculum"]["required_consecutive_evaluations"] == 2
    assert config["output_dir"] != old["output_dir"]
    assert not config.get("continuation") and not config.get("initial_checkpoint")
    for key in ("network", "ppo", "normalization", "env_config", "seed", "device", "torch_num_threads"):
        assert config[key] == v08[key], key
    for key in ("rollout_steps", "log_interval_updates", "checkpoint_interval_updates", "capture_rate_window"):
        assert config["training"][key] == v08["training"][key], key
    validate_config(config)


def test_zero_resets_are_fixed_without_sampling_and_preserve_other_initial_state(short_config):
    reference = PursuitEnv(short_config["env_config"])
    reference.reset(seed=17)
    env = DirectionCurriculumEnv(short_config["env_config"], 0.0, sampling_seed=17)
    before = deepcopy(env.angle_rng.bit_generator.state)
    for _ in range(10):
        observations, info = env.reset()
        assert env.current_angle_degrees == 0.0 and info["arrival_azimuth_offset_degrees"] == 0.0
        np.testing.assert_array_equal(env.states, reference.states)
        np.testing.assert_array_equal(observations, reference._observations())
        np.testing.assert_array_equal(env.accelerations, np.zeros((4, 3)))
        assert env.config == reference.config
        env.step(np.ones((3, 3)))
    _equal(env.angle_rng.bit_generator.state, before)


def test_positive_stage_keeps_continuous_uniform_sampling(short_config):
    env = DirectionCurriculumEnv(short_config["env_config"], 2.0, sampling_seed=17)
    actual = []
    for _ in range(20):
        env.reset()
        actual.append(env.current_angle_degrees)
    expected = np.random.default_rng(17).uniform(-2.0, 2.0, size=20)
    np.testing.assert_array_equal(actual, expected)
    assert len(set(actual)) == 20
    grid = evaluation_angles(2.0, 100)
    assert all(not np.any(grid == value) for value in actual)


def test_stage_settings_use_one_zero_scene_and_one_hundred_positive_scenes():
    curriculum = _yaml(ROOT / "configs/mappo_direction_curriculum_zero_start.yaml")["curriculum"]
    assert stage_evaluation_settings(curriculum, 0.0) == (1, 1.0)
    np.testing.assert_array_equal(evaluation_angles(0.0, 1), np.array([0.0]))
    for half_angle in ROUTE[1:]:
        assert stage_evaluation_settings(curriculum, half_angle) == (100, 0.8)
        np.testing.assert_array_equal(evaluation_angles(half_angle, 100), np.linspace(-half_angle, half_angle, 100))


@pytest.mark.parametrize("count", [0, 2, 100])
def test_zero_evaluation_grid_refuses_repeated_or_empty_zero_scenes(count):
    with pytest.raises(ValueError):
        evaluation_angles(0.0, count)


@pytest.mark.parametrize("field,value", [
    ("zero_angle_evaluation_episodes", 100),
    ("zero_angle_promotion_capture_rate", 0.8),
    ("zero_angle_evaluation_episodes", True),
])
def test_invalid_zero_protocol_is_rejected_before_creating_output(short_config, field, value):
    from scripts.train_direction_curriculum import validate_config

    config = deepcopy(short_config)
    config["curriculum"][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def _updated_agent(config):
    set_seed(31, 1)
    agent = MAPPOAgent(config)
    env = DirectionCurriculumEnv(config["env_config"], 0.0, sampling_seed=7)
    observations, info = env.reset(seed=7)
    buffer = RolloutBuffer(6, 3, 21, 24, 3)
    for _ in range(6):
        actions, log_probs, raw_actions, value = agent.act(observations, info["global_state"])
        next_observations, rewards, terminated, truncated, next_info = env.step(actions)
        buffer.add(observations, info["global_state"], raw_actions, log_probs, float(rewards[0]), value,
                   0.0 if terminated else agent.value(next_info["global_state"]), terminated, truncated)
        observations, info = next_observations, next_info
        if terminated or truncated:
            observations, info = env.reset()
    buffer.compute_returns(config["ppo"]["gamma"], config["ppo"]["gae_lambda"])
    agent.update(buffer)
    return agent


def test_real_zero_evaluation_records_one_complete_fixed_episode_without_training_mutation(
    tmp_path, short_config
):
    from scripts.train_direction_curriculum import evaluate_policy

    agent = _updated_agent(short_config)
    models = {name: deepcopy(getattr(agent, name).state_dict()) for name in MODEL_KEYS}
    assert models["actor_optimizer"]["state"] and models["critic_optimizer"]["state"]
    before = (random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone())
    output = tmp_path / "evaluation"
    summary = evaluate_policy(agent, short_config["env_config"], 0.0, 1, 1000, output)
    assert summary["episodes"] == 1 and summary["capture_rate"] in (0.0, 1.0)
    assert _json(output / "angles.json") == [0.0]
    row = _rows(output / "evaluation_metrics.csv")
    assert len(row) == 1 and row[0]["seed"] == "1000" and float(row[0]["angle_degrees"]) == 0.0
    with np.load(output / "first_episode.npz", allow_pickle=False) as data:
        assert data["states"].shape == (summary["average_episode_length"] + 1, 4, 6)
        assert np.isfinite(data["states"]).all()
        np.testing.assert_array_equal(data["positions"][0, -1], _yaml(short_config["env_config"])["initialization"]["attacker_position"])
    for name in MODEL_KEYS:
        _equal(getattr(agent, name).state_dict(), models[name])
    _equal(random.getstate(), before[0])
    _equal(np.random.get_state(), before[1])
    _equal(torch.get_rng_state(), before[2])
    assert agent.actor.training and agent.critic.training


def test_one_zero_success_at_budget_does_not_promote(tmp_path, short_config):
    calls = []

    def evaluate(agent, env_path, half_angle, episodes, seed, output_dir):
        calls.append((half_angle, episodes))
        return _summary(cap=0.0 if len(calls) == 1 else 1.0, episodes=episodes)

    output, result = _train(tmp_path, short_config, evaluate)
    assert calls == [(0.0, 1), (0.0, 1)]
    assert result["status"] == "stopped_stage_budget" and result["env_steps"] == 8
    assert len(result["stages"]) == 1 and result["stages"][0]["consecutive_passes"] == 1
    assert not (output / "stage_02_pm2").exists()
    assert _json(output / "stage_01_pm0/evaluation_angles.json") == [0.0]


def test_zero_failure_resets_consecutive_success_streak(tmp_path, short_config):
    config = deepcopy(short_config)
    config["curriculum"]["max_stage_steps"] = 16
    config["training"]["total_env_steps"] = 16 * len(ROUTE)
    outcomes = iter([1.0, 0.0, 1.0, 0.0])

    def evaluate(agent, env_path, half_angle, episodes, seed, output_dir):
        return _summary(cap=next(outcomes), episodes=episodes)

    output, result = _train(tmp_path, config, evaluate)
    assert result["env_steps"] == 16 and len(result["stages"]) == 1
    history = _rows(output / "evaluation_history.csv")
    assert [int(row["consecutive_passes"]) for row in history] == [1, 0, 1, 0]
    assert [row["passed"] for row in history] == ["True", "False", "True", "False"]
    assert not (output / "stage_02_pm2").exists()


def test_fractional_single_zero_episode_result_is_rejected(tmp_path, short_config):
    def evaluate(agent, env_path, half_angle, episodes, seed, output_dir):
        return _summary(cap=0.8, episodes=episodes)

    with pytest.raises(ValueError):
        _train(tmp_path, short_config, evaluate)
    assert not (tmp_path / "run/stage_02_pm2").exists()


def test_zero_promotes_only_after_two_captures_and_next_stage_inherits_historical_best(
    tmp_path, short_config
):
    counts, calls = {}, []

    def evaluate(agent, env_path, half_angle, episodes, seed, output_dir):
        stage = _stage_number(output_dir)
        calls.append((stage, half_angle, episodes))
        if stage == 1:
            counts[stage] = counts.get(stage, 0) + 1
            return _summary(cap=1.0, episodes=episodes, capture_time=1.0 if counts[stage] == 1 else 2.0)
        # Excellent previous-zero monitoring cannot promote failing +/-2.
        return _summary(cap=1.0 if half_angle == 0.0 else 0.0, episodes=episodes)

    output, result = _train(tmp_path, short_config, evaluate)
    assert result["env_steps"] == 16 and len(result["stages"]) == 2
    assert calls == [(1, 0.0, 1), (1, 0.0, 1), (2, 2.0, 4), (2, 0.0, 1), (2, 2.0, 4), (2, 0.0, 1)]
    old = output / "stage_01_pm0/checkpoints"
    best, last = _payload(old / "best_eval.pt"), _payload(old / "last.pt")
    initial = _payload(output / "stage_02_pm2/checkpoints/initial.pt")
    assert best["training_state"]["stage_env_steps"] == 4 and last["training_state"]["stage_env_steps"] == 8
    assert any(not torch.equal(best["actor"][key], last["actor"][key]) for key in best["actor"])
    for name in MODEL_KEYS:
        _equal(initial[name], best[name])
    assert best["actor_optimizer"]["state"] and best["critic_optimizer"]["state"]
    assert result["stages"][1]["parent_checkpoint_sha256"] == hashlib.sha256((old / "best_eval.pt").read_bytes()).hexdigest()
    assert result["stages"][1]["inherited_checkpoint_env_steps"] == 4
    assert initial["training_state"]["env_steps"] == 8
    assert not (output / "stage_03_pm5").exists()

    fresh_state = _payload(old / "initial.pt")
    assert fresh_state["training_state"]["env_steps"] == 0
    assert not fresh_state["actor_optimizer"]["state"] and not fresh_state["critic_optimizer"]["state"]
    set_seed(short_config["seed"], short_config["torch_num_threads"])
    fresh = MAPPOAgent(short_config)
    _equal(fresh_state["actor"], fresh.actor.state_dict())
    _equal(fresh_state["critic"], fresh.critic.state_dict())


def test_all_seven_stages_keep_current_gate_independent_of_previous_failure(tmp_path, short_config):
    current_counts, monitor_calls = {}, []

    def evaluate(agent, env_path, half_angle, episodes, seed, output_dir):
        stage = _stage_number(output_dir)
        if half_angle != ROUTE[stage - 1]:
            monitor_calls.append((stage, half_angle, episodes))
            return _summary(cap=0.0, episodes=episodes)
        current_counts[stage] = current_counts.get(stage, 0) + 1
        cap = 1.0 if stage == 1 else 0.8
        return _summary(cap=cap, episodes=episodes,
                        capture_time=1.0 if current_counts[stage] == 1 else 2.0)

    output, result = _train(tmp_path, short_config, evaluate)
    assert result["status"] == "completed" and result["env_steps"] == 56
    assert [stage["half_angle_degrees"] for stage in result["stages"]] == ROUTE
    assert [stage["stage_env_steps"] for stage in result["stages"]] == [8] * 7
    assert current_counts == {number: 2 for number in range(1, 8)}
    expected = [(stage, ROUTE[stage-2], 1 if stage == 2 else 4) for stage in range(2, 8) for _ in range(2)]
    assert monitor_calls == expected
    for stage in range(1, 7):
        previous = output / "stage_{:02d}_pm{:g}/checkpoints".format(stage, ROUTE[stage-1])
        next_path = output / "stage_{:02d}_pm{:g}/checkpoints/initial.pt".format(stage+1, ROUTE[stage])
        selected, inherited = _payload(previous / "best_eval.pt"), _payload(next_path)
        for key in MODEL_KEYS:
            _equal(inherited[key], selected[key])
    for row in _rows(output / "evaluation_history.csv"):
        if row["role"] == "previous":
            assert row["passed"] == "" and row["consecutive_passes"] == ""
