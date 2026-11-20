"""Direction curriculum geometry, selection, optimizer handoff and short flows.

Training fixtures use at most 32 environment steps and small networks. They are
engineering checks, not v09 training runs or evidence of task performance.
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
    DirectionCurriculumEnv, PromotionGate, arrival_position, evaluation_angles,
    evaluation_key, load_best_eval_checkpoint,
)
from utils.experiment import set_seed


ROOT = Path(__file__).resolve().parents[1]


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


@pytest.fixture
def v08_environment():
    return _yaml(ROOT / "configs/baseline_capture500.yaml")


@pytest.fixture
def physical_path(tmp_path, v08_environment):
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(v08_environment, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def small_learning_config():
    config = _yaml(ROOT / "configs/mappo_capture500.yaml")
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["training"]["rollout_steps"] = 8
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 4
    return config


def _rollout(agent, config_path, steps=8):
    env = PursuitEnv(config_path)
    observations, info = env.reset(seed=17)
    buffer = RolloutBuffer(steps, 3, 21, 24, 3)
    for _ in range(steps):
        actions, log_probs, raw_actions, value = agent.act(observations, info["global_state"])
        next_observations, reward, terminated, truncated, next_info = env.step(actions)
        next_value = 0.0 if terminated else agent.value(next_info["global_state"])
        buffer.add(observations, info["global_state"], raw_actions, log_probs,
                   float(reward[0]), value, next_value, terminated, truncated)
        observations, info = next_observations, next_info
        if terminated or truncated:
            observations, info = env.reset()
    buffer.compute_returns(agent.config["ppo"]["gamma"], agent.config["ppo"]["gae_lambda"])
    return buffer


@pytest.mark.parametrize("angle", [0.0, 10.0, -30.0, 60.0, 90.0, -90.0])
def test_arrival_rotation_preserves_base_distance_and_height(v08_environment, angle):
    before = deepcopy(v08_environment)
    base = np.asarray(before["mission"]["base_position"])
    original = np.asarray(before["initialization"]["attacker_position"])
    position = arrival_position(v08_environment, angle)
    theta = np.deg2rad(angle)
    relative = original - base
    expected = base + np.array([
        np.cos(theta) * relative[0] - np.sin(theta) * relative[1],
        np.sin(theta) * relative[0] + np.cos(theta) * relative[1], relative[2],
    ])
    np.testing.assert_allclose(position, expected, rtol=0, atol=1e-12)
    assert position[2] == original[2]
    assert np.linalg.norm(position - base) == pytest.approx(500.0, abs=1e-10)
    assert np.isfinite(position).all()
    assert v08_environment == before


def test_arrival_rotation_is_about_actual_base_not_world_origin(v08_environment):
    config = deepcopy(v08_environment)
    config["mission"]["base_position"] = [-10.0, 12.0, 7.0]
    config["initialization"]["attacker_position"] = [190.0, 112.0, 50.0]
    rotated = arrival_position(config, 90.0)
    np.testing.assert_allclose(rotated, [-110.0, 212.0, 50.0], atol=1e-12)


def test_training_angles_are_continuous_uniform_and_seeded(physical_path):
    first = DirectionCurriculumEnv(physical_path, 30.0, sampling_seed=17)
    second = DirectionCurriculumEnv(physical_path, 30.0, sampling_seed=17)
    sampled = []
    for _ in range(20):
        first.reset()
        second.reset()
        sampled.append(first.current_angle_degrees)
        assert first.current_angle_degrees == second.current_angle_degrees
    expected = np.random.default_rng(17).uniform(-30.0, 30.0, size=20)
    np.testing.assert_array_equal(sampled, expected)
    assert len(set(sampled)) == 20
    assert np.all(np.abs(sampled) <= 30.0)
    grid = evaluation_angles(30.0, 100)
    assert all(np.min(np.abs(grid - value)) > 1e-9 for value in sampled)
    first.reset(seed=101)
    repeated = first.current_angle_degrees
    first.reset(seed=101)
    assert first.current_angle_degrees == repeated


def test_curriculum_reset_changes_only_attacker_position(physical_path, v08_environment):
    env = DirectionCurriculumEnv(physical_path, 90.0, sampling_seed=9)
    reference = PursuitEnv(physical_path)
    original_config = deepcopy(env.config)
    for _ in range(5):
        observations, info = env.reset()
        expected = reference._initial_states.copy()
        expected[-1, :3] = arrival_position(v08_environment, env.current_angle_degrees)
        np.testing.assert_array_equal(env.states, expected)
        np.testing.assert_array_equal(env.accelerations, np.zeros((4, 3)))
        assert observations.shape == (3, 21) and info["global_state"].shape == (24,)
        assert env.config == original_config
        assert env.reward_coefficients == reference.reward_coefficients
        env.step(np.ones((3, 3)))


def test_fixed_evaluation_angle_is_unaffected_by_episode_seed(physical_path):
    env = DirectionCurriculumEnv(physical_path, 30.0, sampling_seed=1, fixed_angle_degrees=12.0)
    first, _ = env.reset(seed=1000)
    env.step(np.ones((3, 3)))
    second, _ = env.reset(seed=1099)
    assert env.current_angle_degrees == 12.0
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(env.accelerations, np.zeros((4, 3)))


@pytest.mark.parametrize("half_angle", [10.0, 30.0, 60.0, 90.0])
def test_evaluation_grid_is_fixed_100_angles_covering_endpoints(half_angle):
    angles = evaluation_angles(half_angle, 100)
    np.testing.assert_array_equal(angles, np.linspace(-half_angle, half_angle, 100))
    assert angles[0] == -half_angle and angles[-1] == half_angle
    assert len(np.unique(angles)) == 100
    np.testing.assert_allclose(np.diff(angles), 2 * half_angle / 99, atol=1e-12)
    np.testing.assert_array_equal(angles, evaluation_angles(half_angle, 100))


def _summary(cap=0.8, failure=None, time=17.0, distance=4.0, collision=0.1, episodes=4):
    if failure is None:
        failure = 1.0 - cap
    return dict(episodes=episodes, capture_rate=cap, enemy_attack_success_rate=failure,
                average_capture_time=None if cap == 0 else time,
                average_min_distance=distance, collision_episode_rate=collision)


@pytest.mark.parametrize("field,value", [
    ("capture_rate", 0.9),
    ("enemy_attack_success_rate", 0.1),
    ("average_capture_time", 16.0),
    ("average_min_distance", 3.0),
    ("collision_episode_rate", 0.0),
])
def test_best_eval_uses_each_requested_tie_breaker(field, value):
    baseline = _summary()
    better = dict(baseline, **{field: value})
    assert evaluation_key(better) > evaluation_key(baseline)
    assert baseline == _summary()


def test_best_eval_criteria_are_lexicographic_and_ignore_reward():
    higher_capture = _summary(cap=0.9, failure=0.1, time=30.0, distance=100.0, collision=1.0)
    lower_capture = _summary(cap=0.8, failure=0.0, time=1.0, distance=0.1, collision=0.0)
    assert evaluation_key(higher_capture) > evaluation_key(lower_capture)
    same_capture_less_failure = _summary(failure=0.0, time=100.0, distance=100.0, collision=1.0)
    assert evaluation_key(same_capture_less_failure) > evaluation_key(_summary())
    assert evaluation_key(dict(_summary(), average_episode_reward=1e9)) == evaluation_key(_summary())
    assert evaluation_key(_summary()) == evaluation_key(deepcopy(_summary()))


def test_no_capture_keeps_null_time_and_finite_comparison_key():
    summary = _summary(cap=0.0)
    assert summary["average_capture_time"] is None
    assert np.isfinite(evaluation_key(summary)).all()
    assert summary["average_capture_time"] is None
    assert evaluation_key(_summary(cap=0.01)) > evaluation_key(summary)
    with pytest.raises(ValueError):
        evaluation_key(dict(_summary(), average_capture_time=None))


@pytest.mark.parametrize("field,value", [
    ("capture_rate", float("nan")),
    ("capture_rate", 1.1),
    ("average_capture_time", float("inf")),
    ("average_min_distance", -1.0),
    ("collision_episode_rate", -0.1),
])
def test_nonfinite_or_invalid_eval_selection_is_rejected(field, value):
    with pytest.raises(ValueError):
        evaluation_key(dict(_summary(), **{field: value}))


def test_promotion_requires_two_consecutive_current_stage_passes():
    gate = PromotionGate(threshold=0.8, required_consecutive=2)
    assert not gate.observe(0.8) and gate.consecutive == 1
    assert not gate.observe(0.79) and gate.consecutive == 0
    assert not gate.observe(1.0) and gate.consecutive == 1
    assert gate.observe(0.8) and gate.consecutive == 2


def test_best_checkpoint_restores_weights_both_adam_moments_and_rng(
    tmp_path, physical_path, small_learning_config
):
    set_seed(31, 1)
    agent = MAPPOAgent(small_learning_config)
    agent.update(_rollout(agent, physical_path))
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    best = tmp_path / "best_eval.pt"
    agent.save(best, training_state={"env_steps": 8})
    expected = _payload(best)
    digest = hashlib.sha256(best.read_bytes()).hexdigest()
    expected_random = random.random()
    expected_numpy = np.random.random(3)
    expected_torch = torch.rand(3)
    agent.update(_rollout(agent, physical_path))
    assert any(not torch.equal(value, expected["actor"][key])
               for key, value in agent.actor.state_dict().items())
    restored, payload = load_best_eval_checkpoint(best, small_learning_config, device="cpu")
    assert restored.actor.training and restored.critic.training
    for key, actual in (("actor", restored.actor.state_dict()), ("critic", restored.critic.state_dict()),
                        ("actor_optimizer", restored.actor_optimizer.state_dict()),
                        ("critic_optimizer", restored.critic_optimizer.state_dict())):
        _assert_nested_equal(actual, expected[key])
    for optimizer in (restored.actor_optimizer, restored.critic_optimizer):
        assert optimizer.state_dict()["state"]
        assert all({"step", "exp_avg", "exp_avg_sq"} <= set(state)
                   for state in optimizer.state_dict()["state"].values())
    assert random.random() == expected_random
    np.testing.assert_array_equal(np.random.random(3), expected_numpy)
    assert torch.equal(torch.rand(3), expected_torch)
    assert payload["training_state"]["env_steps"] == 8
    assert hashlib.sha256(best.read_bytes()).hexdigest() == digest


def test_handoff_rejects_changed_mappo_hyperparameters(tmp_path, small_learning_config):
    set_seed(12, 1)
    agent = MAPPOAgent(small_learning_config)
    source = tmp_path / "best_eval.pt"
    agent.save(source)
    mismatch = deepcopy(small_learning_config)
    mismatch["ppo"]["actor_learning_rate"] *= 2
    with pytest.raises(ValueError):
        load_best_eval_checkpoint(source, mismatch, device="cpu")


def _short_curriculum_config(tmp_path):
    config = _yaml(ROOT / "configs/mappo_direction_curriculum.yaml")
    config["env_config"] = str(ROOT / "configs/baseline_capture500.yaml")
    config["output_dir"] = str(tmp_path / "run")
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 2
    config["training"]["rollout_steps"] = 3
    config["curriculum"]["max_stage_steps"] = 8
    config["curriculum"]["evaluation_interval_steps"] = 4
    config["curriculum"]["evaluation_episodes"] = 4
    config["training"]["total_env_steps"] = 8 * len(config["curriculum"]["stage_half_angles_degrees"])
    path = tmp_path / "curriculum.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, config


def _stage_number(output_dir):
    part = next(part for part in Path(output_dir).parts if part.startswith("stage_") and "_pm" in part)
    return int(part.split("_")[1])


def test_short_curriculum_stops_failed_stage_without_entering_next(tmp_path):
    from scripts.train_direction_curriculum import train_curriculum

    config_path, _ = _short_curriculum_config(tmp_path)
    calls = []

    def evaluate(agent, env_config_path, half_angle_degrees, episodes, seed, output_dir):
        calls.append((half_angle_degrees, episodes))
        return _summary(cap=0.8 if len(calls) == 1 else 0.79, episodes=episodes)

    result = train_curriculum(config_path, evaluation_fn=evaluate)
    assert result["status"] == "stopped_stage_budget"
    assert result["env_steps"] == 8 and len(result["stages"]) == 1
    assert calls == [(10.0, 4), (10.0, 4)]
    assert result["stages"][0]["consecutive_passes"] == 0
    assert not (tmp_path / "run/stage_02_pm30").exists()


def test_all_stages_use_historical_best_and_preserve_complete_adam_state(tmp_path):
    from scripts.train_direction_curriculum import train_curriculum

    config_path, config = _short_curriculum_config(tmp_path)
    angles = config["curriculum"]["stage_half_angles_degrees"]
    current_counts = {}
    monitor_calls = []

    def evaluate(agent, env_config_path, half_angle_degrees, episodes, seed, output_dir):
        stage = _stage_number(output_dir)
        if half_angle_degrees != angles[stage - 1]:
            monitor_calls.append((stage, half_angle_degrees))
            # Catastrophic-forgetting signal must not block promotion or pick best.
            return _summary(cap=0.0, episodes=episodes)
        current_counts[stage] = current_counts.get(stage, 0) + 1
        return _summary(cap=1.0 if current_counts[stage] == 1 else 0.8, episodes=episodes)

    result = train_curriculum(config_path, evaluation_fn=evaluate)
    output = tmp_path / "run"
    assert result["status"] == "completed" and result["env_steps"] == 32
    assert len(result["stages"]) == 4
    assert current_counts == {1: 2, 2: 2, 3: 2, 4: 2}
    assert monitor_calls == [(stage, angles[stage - 2]) for stage in (2, 3, 4) for _ in range(2)]
    for stage in range(1, 4):
        previous = output / ("stage_{:02d}_pm{:g}".format(stage, angles[stage - 1])) / "checkpoints"
        following = output / ("stage_{:02d}_pm{:g}".format(stage + 1, angles[stage])) / "checkpoints"
        best, last = _payload(previous / "best_eval.pt"), _payload(previous / "last.pt")
        initial = _payload(following / "initial.pt")
        for key in ("actor", "critic", "actor_optimizer", "critic_optimizer"):
            _assert_nested_equal(initial[key], best[key])
        assert any(not torch.equal(last["actor"][key], best["actor"][key]) for key in best["actor"])
        assert result["stages"][stage]["parent_checkpoint_sha256"] == hashlib.sha256((previous / "best_eval.pt").read_bytes()).hexdigest()
    first = _payload(output / "stage_01_pm10/checkpoints/initial.pt")
    assert not first["actor_optimizer"]["state"] and not first["critic_optimizer"]["state"]
    set_seed(config["seed"], config["torch_num_threads"])
    fresh = MAPPOAgent(config)
    _assert_nested_equal(first["actor"], fresh.actor.state_dict())
    _assert_nested_equal(first["critic"], fresh.critic.state_dict())


def test_previous_stage_success_cannot_promote_failing_current_stage(tmp_path):
    from scripts.train_direction_curriculum import train_curriculum

    config_path, config = _short_curriculum_config(tmp_path)
    angles = config["curriculum"]["stage_half_angles_degrees"]
    calls = []

    def evaluate(agent, env_config_path, half_angle_degrees, episodes, seed, output_dir):
        stage = _stage_number(output_dir)
        calls.append((stage, half_angle_degrees))
        cap = 0.8 if stage == 1 else (1.0 if half_angle_degrees == angles[stage - 2] else 0.79)
        return _summary(cap=cap, episodes=episodes)

    result = train_curriculum(config_path, evaluation_fn=evaluate)
    assert result["status"] == "stopped_stage_budget" and result["env_steps"] == 16
    assert len(result["stages"]) == 2
    assert result["stages"][1]["selected_evaluation"]["capture_rate"] == 0.79
    assert result["stages"][1]["consecutive_passes"] == 0
    assert not (tmp_path / "run/stage_03_pm60").exists()


def test_deterministic_stage_evaluation_does_not_mutate_training_state(
    tmp_path, small_learning_config, v08_environment
):
    from scripts.train_direction_curriculum import evaluate_policy

    config = deepcopy(v08_environment)
    config["environment"]["max_episode_steps"] = 2
    env_path = tmp_path / "short_environment.yaml"
    env_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    set_seed(17, 1)
    agent = MAPPOAgent(small_learning_config)
    agent.update(_rollout(agent, env_path))
    agent.actor.train()
    agent.critic.train()
    weights = {"actor": deepcopy(agent.actor.state_dict()), "critic": deepcopy(agent.critic.state_dict())}
    optimizers = {
        "actor": deepcopy(agent.actor_optimizer.state_dict()),
        "critic": deepcopy(agent.critic_optimizer.state_dict()),
    }
    assert optimizers["actor"]["state"] and optimizers["critic"]["state"]
    before_rng = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    result = evaluate_policy(agent, env_path, 10.0, 4, 1000, tmp_path / "evaluation")
    assert result["episodes"] == 4 and result["capture_rate"] == 0.0
    assert result["average_capture_time"] is None
    assert agent.actor.training and agent.critic.training
    assert all(p.requires_grad for n in (agent.actor, agent.critic) for p in n.parameters())
    _assert_nested_equal(agent.actor.state_dict(), weights["actor"])
    _assert_nested_equal(agent.critic.state_dict(), weights["critic"])
    _assert_nested_equal(agent.actor_optimizer.state_dict(), optimizers["actor"])
    _assert_nested_equal(agent.critic_optimizer.state_dict(), optimizers["critic"])
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert random.getstate() == before_python
    _assert_nested_equal(np.random.get_state(), before_numpy)


@pytest.fixture
def entrypoint_registry(tmp_path, small_learning_config):
    config_path = tmp_path / "configs/training.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(small_learning_config), encoding="utf-8")
    module_path = tmp_path / "scripts/train_direction_curriculum.py"
    module_path.parent.mkdir()
    module_path.write_text('"""Existing curriculum training module."""\n', encoding="utf-8")
    return {
        "schema_version": 1,
        "current_version": "v09",
        "versions": [
            {"id": "v08", "training_config_path": "configs/training.yaml"},
            {"id": "v09", "training_config_path": "configs/training.yaml",
             "training_entrypoint": "scripts.train_direction_curriculum"},
        ],
        "runs": [], "events": [], "models": [],
    }


def test_v09_experiment_command_routes_to_curriculum_without_creating_run(
    tmp_path, entrypoint_registry
):
    from scripts.experiment_index import suggest_next_run, validate_registry

    assert validate_registry(entrypoint_registry, tmp_path) == []
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    proposed = suggest_next_run(entrypoint_registry, "v09", seed=7,
                                date_string="20261109", root=tmp_path)
    assert proposed["id"] == "v09-r01"
    assert proposed["command"] == (
        "& .\\.venv\\Scripts\\python.exe -m scripts.train_direction_curriculum "
        "--config configs/training.yaml --seed 7 --output-dir " + proposed["path"]
    )
    assert "scripts.train_mappo" not in proposed["command"]
    assert not (tmp_path / proposed["path"]).exists()
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*")} == before


def test_legacy_experiment_without_entrypoint_keeps_default_training_command(
    tmp_path, entrypoint_registry
):
    from scripts.experiment_index import suggest_next_run, validate_registry

    entrypoint_registry["current_version"] = "v08"
    assert validate_registry(entrypoint_registry, tmp_path) == []
    proposed = suggest_next_run(entrypoint_registry, "v08", date_string="20261109", root=tmp_path)
    assert " -m scripts.train_mappo --config configs/training.yaml " in proposed["command"]
    assert "train_direction_curriculum" not in proposed["command"]
    assert suggest_next_run(entrypoint_registry, "v09", root=tmp_path)["command"] is None


@pytest.mark.parametrize("module", [None, 9, "scripts.train_direction_curriculum.extra",
                                    "../train_direction_curriculum",
                                    "scripts.train_mappo; unwanted_command"])
def test_explicit_invalid_training_entrypoint_is_rejected(tmp_path, entrypoint_registry, module):
    from scripts.experiment_index import suggest_next_run, validate_registry

    entrypoint_registry["versions"][1]["training_entrypoint"] = module
    assert any("training_entrypoint" in error
               for error in validate_registry(entrypoint_registry, tmp_path))
    with pytest.raises(ValueError, match="training_entrypoint"):
        suggest_next_run(entrypoint_registry, "v09", root=tmp_path)


def test_explicit_missing_training_module_is_rejected(tmp_path, entrypoint_registry):
    from scripts.experiment_index import suggest_next_run, validate_registry

    entrypoint_registry["versions"][1]["training_entrypoint"] = "scripts.missing_curriculum"
    assert any("module does not exist" in error
               for error in validate_registry(entrypoint_registry, tmp_path))
    with pytest.raises(ValueError, match="training_entrypoint module does not exist"):
        suggest_next_run(entrypoint_registry, "v09", root=tmp_path)
