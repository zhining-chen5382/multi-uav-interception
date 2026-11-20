"""Tiny PPO checks for the fixed-scene optimizer-state retention comparison.

These fixtures exercise actual PPO and deterministic episodes at a shortened
horizon. Their outcomes do not measure the real checkpoint's task performance.
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


ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ("actor", "critic", "actor_optimizer", "critic_optimizer", "rng_state")


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


@pytest.fixture
def retention_config(tmp_path, monkeypatch):
    from scripts import train_mappo, train_policy_retention

    monkeypatch.setattr(train_mappo, "run_post_training_analysis", lambda *args: {"status": "disabled"})
    monkeypatch.setattr(train_policy_retention, "run_post_training_analysis", lambda *args: {"status": "disabled"})
    config = _yaml(ROOT / "configs/mappo_capture500.yaml")
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
    config["training"]["checkpoint_interval_updates"] = 1
    config["training"]["total_env_steps"] = 5
    config["output_dir"] = str(tmp_path / "source")
    source_config = tmp_path / "source.yaml"
    source_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    train_mappo.train(source_config)
    checkpoint = tmp_path / "source/checkpoints/final.pt"
    saved = _payload(checkpoint)
    assert saved["actor_optimizer"]["state"] and saved["critic_optimizer"]["state"]
    assert saved["training_state"]["partial_episode"]["episode_length"] == 2
    config["training"]["total_env_steps"] = 13
    config["output_dir"] = str(tmp_path / "retention")
    config["retention"] = dict(checkpoint=str(checkpoint), expected_sha256=_sha(checkpoint),
        groups=[dict(name="A", reset_adam=False), dict(name="B", reset_adam=True)],
        evaluation_interval_steps=4, evaluation_seed=1000)
    return config


def _context(config):
    from scripts.train_policy_retention import validate_config

    return validate_config(config, _yaml(config["env_config"]))


def _group(tmp_path, config, *, reset=False, name=None):
    from scripts.train_policy_retention import run_group

    name = name or ("B" if reset else "A")
    output = tmp_path / name
    result = run_group(deepcopy(config), Path(config["env_config"]), _context(config),
        dict(name="B" if reset else "A", reset_adam=reset), output)
    return output, result


def test_initialization_only_resets_adam_state_in_group_b(retention_config):
    from scripts.train_policy_retention import initialize_agent

    source = _context(retention_config)
    original = _payload(retention_config["retention"]["checkpoint"])
    a, _ = initialize_agent(retention_config, source, False)
    a_rng = dict(python=random.getstate(), numpy=deepcopy(np.random.get_state()), torch=torch.get_rng_state().clone())
    b, _ = initialize_agent(retention_config, source, True)
    for name in ("actor", "critic"):
        _equal(getattr(a, name).state_dict(), original[name])
        _equal(getattr(b, name).state_dict(), original[name])
    for name in ("actor_optimizer", "critic_optimizer"):
        _equal(getattr(a, name).state_dict(), original[name])
        empty = getattr(b, name).state_dict()
        assert not empty["state"]
        _equal(empty["param_groups"], original[name]["param_groups"])
    for actual, key in ((random.getstate(), "python"), (np.random.get_state(), "numpy"),
                        (torch.get_rng_state(), "torch")):
        _equal(actual, a_rng[key])
        _equal(actual, original["rng_state"][key])


def test_both_groups_have_identical_initial_models_rng_and_first_training_rollout(tmp_path, retention_config):
    original_path = Path(retention_config["retention"]["checkpoint"])
    digest = _sha(original_path)
    a, _ = _group(tmp_path, retention_config)
    b, _ = _group(tmp_path, retention_config, reset=True)
    initial_a, initial_b = _payload(a / "checkpoints/initial.pt"), _payload(b / "checkpoints/initial.pt")
    for key in ("actor", "critic", "rng_state"):
        _equal(initial_a[key], initial_b[key])
    assert initial_a["actor_optimizer"]["state"] and initial_a["critic_optimizer"]["state"]
    assert not initial_b["actor_optimizer"]["state"] and not initial_b["critic_optimizer"]["state"]
    for initial in (initial_a, initial_b):
        state = initial["training_state"]
        assert state["env_steps"] == state["update"] == state["episodes"] == 0
        assert state["environment_step_count"] == 0
    with np.load(a / "first_rollout.npz", allow_pickle=False) as left, np.load(b / "first_rollout.npz", allow_pickle=False) as right:
        assert set(left.files) == set(right.files)
        assert {"observations", "states", "raw_actions", "rewards", "values"} <= set(left.files)
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
    final_a, final_b = _payload(a / "checkpoints/final.pt"), _payload(b / "checkpoints/final.pt")
    assert any(not torch.equal(final_a["actor"][key], final_b["actor"][key]) for key in final_a["actor"])
    assert _sha(original_path) == digest


def test_exact_evaluations_keep_full_ppo_batches_and_evaluate_final_tail_after_update(tmp_path, retention_config):
    output, _ = _group(tmp_path, retention_config)
    updates, evaluations = _rows(output / "updates.csv"), _rows(output / "evaluations.csv")
    assert [int(row["env_steps"]) for row in updates] == [3, 6, 9, 12, 13]
    assert [int(row["rollout_steps"]) for row in updates] == [3, 3, 3, 3, 1]
    assert [int(row["env_steps"]) for row in evaluations] == [0, 4, 8, 12, 13]
    assert [int(row["last_update_env_steps"]) for row in evaluations] == [0, 3, 6, 12, 13]
    assert [int(row["pending_rollout_steps"]) for row in evaluations] == [0, 1, 2, 0, 0]
    assert [int(row["update"]) for row in evaluations] == [0, 1, 2, 4, 5]
    assert evaluations[-1]["checked_after_update"] == "True"
    saved = _payload(output / "checkpoints/final.pt")
    assert saved["training_state"]["env_steps"] == 13
    assert saved["training_state"]["update"] == 5
    for row in evaluations:
        archive = Path(row["trajectory"])
        with np.load(archive, allow_pickle=False) as data:
            metadata = json.loads(data["metadata_json"].item())
            assert data["states"].shape == (metadata["metrics"]["episode_length"] + 1, 4, 6)
            assert np.isfinite(data["states"]).all()


def test_more_frequent_evaluation_does_not_change_training_rng_environment_or_updates(tmp_path, retention_config):
    first, _ = _group(tmp_path, retention_config, name="cadence4")
    alternative = deepcopy(retention_config)
    alternative["retention"]["evaluation_interval_steps"] = 5
    second, _ = _group(tmp_path, alternative, name="cadence5")
    a, b = _payload(first / "checkpoints/final.pt"), _payload(second / "checkpoints/final.pt")
    for key in MODEL_KEYS:
        _equal(a[key], b[key])
    for key in ("env_steps", "update", "episodes", "partial_episode", "environment_state",
                "environment_accelerations", "environment_step_count", "environment_rng_state"):
        _equal(a["training_state"][key], b["training_state"][key])
    columns = ("env_steps", "actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
    assert [{key:row[key] for key in columns} for row in _rows(first / "updates.csv")] == [
        {key:row[key] for key in columns} for row in _rows(second / "updates.csv")]


def test_training_metrics_record_original_learning_rates_and_finite_ppo_diagnostics(tmp_path, retention_config):
    output, _ = _group(tmp_path, retention_config)
    updates = _rows(output / "updates.csv")
    for row in updates:
        assert float(row["actor_learning_rate"]) == retention_config["ppo"]["actor_learning_rate"]
        assert float(row["critic_learning_rate"]) == retention_config["ppo"]["critic_learning_rate"]
        for key in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                    "actor_grad_norm", "critic_grad_norm"):
            assert np.isfinite(float(row[key]))
        assert 0 <= float(row["clip_fraction"]) <= 1


@pytest.mark.parametrize("change", ["hash", "environment", "ppo", "network", "missing_checkpoint", "missing_adam"])
def test_incompatible_source_is_rejected_before_output_is_created(tmp_path, retention_config, change):
    from scripts.train_policy_retention import run_experiment

    config = deepcopy(retention_config)
    if change == "hash":
        config["retention"]["expected_sha256"] = "0" * 64
    elif change == "environment":
        path = tmp_path / "changed_environment.yaml"
        environment = _yaml(config["env_config"])
        environment["reward"]["capture_reward"] += 1
        path.write_text(yaml.safe_dump(environment), encoding="utf-8")
        config["env_config"] = str(path)
    elif change == "ppo":
        config["ppo"]["actor_learning_rate"] *= 0.5
    elif change == "network":
        config["network"]["actor_hidden_sizes"] = [17, 17]
    elif change == "missing_checkpoint":
        config["retention"]["checkpoint"] = str(tmp_path / "absent.pt")
    else:
        path = tmp_path / "missing_adam.pt"
        source = _payload(config["retention"]["checkpoint"])
        source["actor_optimizer"]["state"] = {}
        torch.save(source, path)
        config["retention"].update(checkpoint=str(path), expected_sha256=_sha(path))
    output = tmp_path / "must_not_exist"
    config["output_dir"] = str(output)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        run_experiment(path)
    assert not output.exists()


def test_formal_diagnostic_configuration_changes_only_optimizer_memory():
    config = _yaml(ROOT / "configs/mappo_policy_retention.yaml")
    original = _yaml(ROOT / "configs/mappo_capture500.yaml")
    for key in ("env_config", "seed", "device", "torch_num_threads", "network", "ppo", "normalization"):
        assert config[key] == original[key]
    assert config["training"]["total_env_steps"] == 20000
    assert config["training"]["rollout_steps"] == original["training"]["rollout_steps"] == 512
    assert config["retention"]["evaluation_interval_steps"] == 2000
    assert config["retention"]["groups"] == [dict(name="A", reset_adam=False), dict(name="B", reset_adam=True)]
    assert "v08_" in config["retention"]["checkpoint"]
    assert config["retention"]["checkpoint"].endswith("checkpoints/update_000180.pt")
    assert not config.get("curriculum")
