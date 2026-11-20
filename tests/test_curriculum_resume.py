"""Exact short-run continuation checks; these are not formal training runs."""

from copy import deepcopy
import csv
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_KEYS = ("actor", "critic", "actor_optimizer", "critic_optimizer")


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


def _hash_tree(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).rglob("*") if path.is_file()}


def _summary(cap=0.0, episodes=4):
    return dict(episodes=episodes, capture_rate=cap, enemy_attack_success_rate=1.0 - cap,
                average_capture_time=None if cap == 0 else 1.0,
                average_min_distance=8.0, collision_episode_rate=0.0)


def _constant_evaluator(agent, env_config_path, half_angle_degrees, episodes, seed, output_dir):
    return _summary(episodes=episodes)


@pytest.fixture
def short_config(tmp_path):
    config = _yaml(ROOT / "configs/mappo_direction_curriculum.yaml")
    environment = _yaml(ROOT / "configs/baseline_capture500.yaml")
    # Four steps stop inside an episode; continuation crosses two resets.
    environment["environment"]["max_episode_steps"] = 5
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text(yaml.safe_dump(environment), encoding="utf-8")
    config["env_config"] = str(environment_path)
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 2
    config["training"]["rollout_steps"] = 3
    config["curriculum"]["max_stage_steps"] = 4
    config["curriculum"]["evaluation_interval_steps"] = 4
    config["curriculum"]["evaluation_episodes"] = 4
    config["training"]["total_env_steps"] = 16
    return config


def _write_config(tmp_path, config, name):
    config = deepcopy(config)
    config["output_dir"] = str(tmp_path / name)
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run(tmp_path, config, name, evaluator=_constant_evaluator):
    from scripts.train_direction_curriculum import train_curriculum

    path = _write_config(tmp_path, config, name)
    result = train_curriculum(path, evaluation_fn=evaluator)
    return tmp_path / name, result


def _continued(config, source, total=12, budgets=None):
    result = deepcopy(config)
    result["continuation"] = {"source_output": str(source)}
    result["curriculum"]["stage_step_budgets"] = [12, 4, 4, 4] if budgets is None else budgets
    result["training"]["total_env_steps"] = total
    return result


def _checkpoint(output, filename="last.pt", stage="stage_01_pm10"):
    return Path(output) / stage / "checkpoints" / filename


@pytest.mark.parametrize("legacy_snapshot", [False, True])
def test_resume_matches_uninterrupted_models_adam_rng_and_partial_environment(
    tmp_path, short_config, legacy_snapshot
):
    source, stopped = _run(tmp_path, short_config, "source")
    assert stopped["env_steps"] == 4
    if legacy_snapshot:
        checkpoint = _checkpoint(source)
        payload = _payload(checkpoint)
        payload["training_state"].pop("environment_internal")
        torch.save(payload, checkpoint)
    source_hashes = _hash_tree(source)
    resumed, result = _run(tmp_path, _continued(short_config, source), "resumed")
    direct_config = deepcopy(short_config)
    direct_config["curriculum"]["stage_step_budgets"] = [12, 4, 4, 4]
    direct_config["training"]["total_env_steps"] = 12
    direct, direct_result = _run(tmp_path, direct_config, "direct")
    assert result["env_steps"] == direct_result["env_steps"] == 12
    actual, expected = _payload(_checkpoint(resumed)), _payload(_checkpoint(direct))
    for key in WEIGHT_KEYS + ("rng_state",):
        _equal(actual[key], expected[key])
    for key in ("env_steps", "update", "episodes", "stage_env_steps", "stage_updates",
                "partial_episode", "recent_episodes", "environment_state",
                "environment_accelerations", "environment_step_count", "environment_rng_state",
                "angle_sampling_rng_state", "current_episode_angle_degrees", "environment_internal"):
        _equal(actual["training_state"][key], expected["training_state"][key])
    assert actual["training_state"]["partial_episode"]["episode_length"] == 2
    assert actual["training_state"]["episodes"] == 2
    initial = _payload(_checkpoint(resumed, "resume_initial.pt"))
    original = _payload(_checkpoint(source))
    for key in WEIGHT_KEYS + ("rng_state",):
        _equal(initial[key], original[key])
    assert initial["training_state"]["env_steps"] == 4
    assert _hash_tree(source) == source_hashes


def test_resume_uses_last_not_best_and_retains_historical_best_candidate(tmp_path, short_config):
    config = deepcopy(short_config)
    config["curriculum"]["max_stage_steps"] = 8
    config["training"]["total_env_steps"] = 32
    calls = []

    def evaluate(agent, env_config_path, half_angle_degrees, episodes, seed, output_dir):
        calls.append(output_dir)
        return _summary(cap=0.5 if len(calls) == 1 else 0.25, episodes=episodes)

    source, result = _run(tmp_path, config, "source", evaluate)
    assert result["env_steps"] == 8
    last, best = _payload(_checkpoint(source)), _payload(_checkpoint(source, "best_eval.pt"))
    assert any(not torch.equal(last["actor"][key], best["actor"][key]) for key in best["actor"])
    original_hashes = _hash_tree(source)
    resumed, result = _run(tmp_path, _continued(config, source), "resumed", evaluate)
    assert result["env_steps"] == 12
    initial = _payload(_checkpoint(resumed, "resume_initial.pt"))
    for key in WEIGHT_KEYS + ("rng_state",):
        _equal(initial[key], last[key])
    selected = _payload(result["selected_checkpoint"])
    for key in WEIGHT_KEYS:
        _equal(selected[key], best[key])
    assert result["stages"][0]["selected_evaluation"]["capture_rate"] == 0.5
    assert selected["training_state"]["env_steps"] == 4
    assert _hash_tree(source) == original_hashes


@pytest.mark.parametrize("change", ["learning_rate", "environment", "angles", "rollout_steps"])
def test_resume_rejects_changed_research_or_training_settings(tmp_path, short_config, change):
    from scripts.train_direction_curriculum import train_curriculum

    source, _ = _run(tmp_path, short_config, "source")
    source_hashes = _hash_tree(source)
    config = _continued(short_config, source)
    if change == "learning_rate":
        config["ppo"]["actor_learning_rate"] *= 2
    elif change == "environment":
        environment = _yaml(config["env_config"])
        environment["reward"]["capture_reward"] += 1
        changed = tmp_path / "changed_environment.yaml"
        changed.write_text(yaml.safe_dump(environment), encoding="utf-8")
        config["env_config"] = str(changed)
    elif change == "angles":
        config["curriculum"]["stage_half_angles_degrees"][0] = 9
    else:
        config["training"]["rollout_steps"] += 1
    path = _write_config(tmp_path, config, "rejected")
    with pytest.raises(ValueError):
        train_curriculum(path, evaluation_fn=_constant_evaluator)
    assert _hash_tree(source) == source_hashes


@pytest.mark.parametrize("total", [0, 4])
def test_resume_rejects_global_cap_at_or_below_consumed_steps(tmp_path, short_config, total):
    from scripts.train_direction_curriculum import train_curriculum

    source, _ = _run(tmp_path, short_config, "source")
    path = _write_config(tmp_path, _continued(short_config, source, total=total), "rejected")
    with pytest.raises(ValueError):
        train_curriculum(path, evaluation_fn=_constant_evaluator)


@pytest.mark.parametrize("corruption", ["missing", "nonfinite", "shape", "internal_step"])
def test_resume_rejects_missing_or_invalid_environment_snapshot(tmp_path, short_config, corruption):
    from scripts.train_direction_curriculum import train_curriculum

    source, _ = _run(tmp_path, short_config, "source")
    modified = tmp_path / "modified_source"
    shutil.copytree(source, modified)
    checkpoint = _checkpoint(modified)
    payload = _payload(checkpoint)
    state = payload["training_state"]
    if corruption == "missing":
        state.pop("environment_state")
    elif corruption == "nonfinite":
        state["environment_state"][0] = float("nan")
    elif corruption == "shape":
        state["environment_accelerations"] = np.zeros((3, 3))
    else:
        state["environment_internal"]["step_count"] = -1
    torch.save(payload, checkpoint)
    path = _write_config(tmp_path, _continued(short_config, modified), "rejected")
    with pytest.raises(ValueError):
        train_curriculum(path, evaluation_fn=_constant_evaluator)


def test_resume_retains_gate_and_best_then_honors_global_cap_after_promotion(tmp_path, short_config):
    def initial_eval(agent, env_config_path, half_angle_degrees, episodes, seed, output_dir):
        return _summary(cap=1.0, episodes=episodes)

    source, stopped = _run(tmp_path, short_config, "source", initial_eval)
    assert stopped["stages"][0]["consecutive_passes"] == 1
    source_best = _payload(_checkpoint(source, "best_eval.pt"))
    source_hashes = _hash_tree(source)
    calls = []

    def evaluate(agent, env_config_path, half_angle_degrees, episodes, seed, output_dir):
        calls.append((half_angle_degrees, Path(output_dir).name))
        if Path(output_dir).name == "previous":
            return _summary(cap=1.0, episodes=episodes)
        return _summary(cap=0.8 if half_angle_degrees == 10 else 0.0, episodes=episodes)

    config = _continued(short_config, source, total=12, budgets=[12, 12, 12, 12])
    resumed, result = _run(tmp_path, config, "resumed", evaluate)
    assert result["env_steps"] == 12
    assert len(result["stages"]) == 2
    assert result["stages"][0]["stage_env_steps"] == 8
    assert result["stages"][0]["consecutive_passes"] == 2
    assert result["stages"][0]["selected_evaluation"]["capture_rate"] == 1.0
    assert result["stages"][1]["stage_env_steps"] == 4
    assert result["stages"][1]["consecutive_passes"] == 0
    assert calls == [(10.0, "current"), (30.0, "current"), (10, "previous")]
    assert not (resumed / "stage_03_pm60").exists()
    inherited = _payload(_checkpoint(resumed, "initial.pt", stage="stage_02_pm30"))
    for key in WEIGHT_KEYS:
        _equal(inherited[key], source_best[key])
    assert inherited["training_state"]["env_steps"] == 8
    with (resumed / "stage_02_pm30/updates.csv").open(encoding="utf-8", newline="") as stream:
        assert max(int(row["env_steps"]) for row in csv.DictReader(stream)) == 12
    assert _hash_tree(source) == source_hashes
