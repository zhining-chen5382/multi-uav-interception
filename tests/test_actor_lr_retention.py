"""Actual tiny PPO checks for a three-arm Actor learning-rate intervention.

Short-horizon fixtures validate engineering isolation, not task performance.
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
RATES = (3e-4, 1e-4, 3e-5)
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
def lr_config(tmp_path, monkeypatch):
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
    source_path = tmp_path / "source_config.yaml"
    source_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    train_mappo.train(source_path)
    checkpoint = tmp_path / "source/checkpoints/final.pt"
    saved = _payload(checkpoint)
    assert saved["actor_optimizer"]["state"] and saved["critic_optimizer"]["state"]
    config["training"]["total_env_steps"] = 13
    config["output_dir"] = str(tmp_path / "lr_triplet")
    config["retention"] = dict(checkpoint=str(checkpoint), expected_sha256=_sha(checkpoint),
        evaluation_interval_steps=4, evaluation_seed=1000,
        groups=[dict(name=name, reset_adam=False, actor_learning_rate=rate)
                for name, rate in zip(("A", "B", "C"), RATES)])
    return config


def _context(config):
    from scripts.train_actor_lr_retention import validate_config

    return validate_config(config, _yaml(config["env_config"]))


@pytest.fixture
def trained_triplet(tmp_path, lr_config):
    from scripts.train_actor_lr_retention import run_experiment

    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(lr_config, sort_keys=False), encoding="utf-8")
    run_experiment(path)
    return Path(lr_config["output_dir"])


def test_initialization_preserves_adam_memory_and_changes_only_actor_group_lr(lr_config):
    from scripts.train_policy_retention import initialize_agent

    source = _context(lr_config)
    original = _payload(lr_config["retention"]["checkpoint"])
    for group in lr_config["retention"]["groups"]:
        config = deepcopy(lr_config)
        config["ppo"]["actor_learning_rate"] = group["actor_learning_rate"]
        agent, _ = initialize_agent(config, source, False,
                                    actor_learning_rate=group["actor_learning_rate"])
        for name in ("actor", "critic"):
            _equal(getattr(agent, name).state_dict(), original[name])
        actor = agent.actor_optimizer.state_dict()
        _equal(actor["state"], original["actor_optimizer"]["state"])
        expected_groups = deepcopy(original["actor_optimizer"]["param_groups"])
        for parameters in expected_groups:
            parameters["lr"] = group["actor_learning_rate"]
        _equal(actor["param_groups"], expected_groups)
        _equal(agent.critic_optimizer.state_dict(), original["critic_optimizer"])
        assert agent.actor_optimizer.param_groups[0]["lr"] == group["actor_learning_rate"]
        assert agent.critic_optimizer.param_groups[0]["lr"] == 1e-3
        for actual, name in ((random.getstate(), "python"), (np.random.get_state(), "numpy"),
                             (torch.get_rng_state(), "torch")):
            _equal(actual, original["rng_state"][name])


def test_three_arms_have_identical_initial_models_rng_and_first_rollout(trained_triplet, lr_config):
    source_path = Path(lr_config["retention"]["checkpoint"])
    assert _sha(source_path) == lr_config["retention"]["expected_sha256"]
    original = _payload(source_path)
    reference_arrays = None
    for name, rate in zip(("A", "B", "C"), RATES):
        directory = trained_triplet / name
        initial = _payload(directory / "checkpoints/initial.pt")
        for key in ("actor", "critic", "rng_state"):
            _equal(initial[key], original[key])
        for key in ("actor_optimizer", "critic_optimizer"):
            _equal(initial[key]["state"], original[key]["state"])
        assert initial["actor_optimizer"]["param_groups"][0]["lr"] == rate
        assert initial["critic_optimizer"]["param_groups"][0]["lr"] == 1e-3
        assert initial["config"]["ppo"]["actor_learning_rate"] == rate
        assert initial["training_state"]["env_steps"] == 0
        assert initial["training_state"]["environment_step_count"] == 0
        with np.load(directory / "first_rollout.npz", allow_pickle=False) as archive:
            arrays = {key:archive[key].copy() for key in archive.files}
        assert len(arrays) == 13
        if reference_arrays is None:
            reference_arrays = arrays
        else:
            for key in reference_arrays:
                np.testing.assert_array_equal(arrays[key], reference_arrays[key])


def test_first_update_critic_is_equal_while_lower_actor_lr_moves_less(trained_triplet):
    first = {name:_payload(trained_triplet / name / "checkpoints/update_000001.pt")
             for name in ("A", "B", "C")}
    for name in ("B", "C"):
        _equal(first[name]["critic"], first["A"]["critic"])
        _equal(first[name]["critic_optimizer"], first["A"]["critic_optimizer"])
    drifts = [float(_rows(trained_triplet / name / "updates.csv")[0]["actor_parameter_delta_l2"])
              for name in ("A", "B", "C")]
    assert 0 < drifts[2] < drifts[1] < drifts[0]


def test_actual_lrs_stay_fixed_and_exact_evaluations_never_split_rollouts(trained_triplet):
    for name, rate in zip(("A", "B", "C"), RATES):
        directory = trained_triplet / name
        updates = _rows(directory / "updates.csv")
        evaluations = _rows(directory / "evaluations.csv")
        assert [int(row["env_steps"]) for row in updates] == [3, 6, 9, 12, 13]
        assert [int(row["rollout_steps"]) for row in updates] == [3, 3, 3, 3, 1]
        assert [int(row["env_steps"]) for row in evaluations] == [0, 4, 8, 12, 13]
        assert [int(row["last_update_env_steps"]) for row in evaluations] == [0, 3, 6, 12, 13]
        assert [int(row["pending_rollout_steps"]) for row in evaluations] == [0, 1, 2, 0, 0]
        assert evaluations[-1]["checked_after_update"] == "True"
        saved = _yaml(directory / "training_config.yaml")
        assert saved["ppo"]["actor_learning_rate"] == rate
        for row in updates:
            assert float(row["actor_learning_rate"]) == rate
            assert float(row["critic_learning_rate"]) == 1e-3
            for key in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
                assert np.isfinite(float(row[key]))
            assert 0 <= float(row["clip_fraction"]) <= 1
        for row in evaluations:
            with np.load(row["trajectory"], allow_pickle=False) as archive:
                metrics = json.loads(archive["metadata_json"].item())["metrics"]
                assert archive["states"].shape == (metrics["episode_length"] + 1, 4, 6)
                assert np.isfinite(archive["states"]).all()


def test_baseline_lr_arm_is_bitwise_equivalent_to_legacy_retention_group(tmp_path, lr_config):
    from scripts.train_policy_retention import run_group

    source = _context(lr_config)
    with_lr, legacy = tmp_path / "with_lr", tmp_path / "legacy"
    run_group(lr_config, Path(lr_config["env_config"]), source,
              lr_config["retention"]["groups"][0], with_lr)
    run_group(lr_config, Path(lr_config["env_config"]), source,
              dict(name="A", reset_adam=False), legacy)
    actual, expected = _payload(with_lr / "checkpoints/final.pt"), _payload(legacy / "checkpoints/final.pt")
    for key in MODEL_KEYS:
        _equal(actual[key], expected[key])
    for key in ("env_steps", "update", "episodes", "partial_episode", "environment_state",
                "environment_accelerations", "environment_step_count", "environment_rng_state"):
        _equal(actual["training_state"][key], expected["training_state"][key])
    assert _rows(with_lr / "updates.csv") == _rows(legacy / "updates.csv")


@pytest.mark.parametrize("change", ["hash", "reward", "critic_lr", "gamma", "network", "missing_adam", "reset_adam"])
def test_non_actor_lr_changes_are_rejected_before_output_creation(tmp_path, lr_config, change):
    from scripts.train_actor_lr_retention import run_experiment

    config = deepcopy(lr_config)
    if change == "hash":
        config["retention"]["expected_sha256"] = "0" * 64
    elif change == "reward":
        path = tmp_path / "changed_environment.yaml"
        environment = _yaml(config["env_config"])
        environment["reward"]["capture_reward"] += 1
        path.write_text(yaml.safe_dump(environment), encoding="utf-8")
        config["env_config"] = str(path)
    elif change == "critic_lr":
        config["ppo"]["critic_learning_rate"] *= 0.5
    elif change == "gamma":
        config["ppo"]["gamma"] *= 0.5
    elif change == "network":
        config["network"]["actor_hidden_sizes"] = [17, 17]
    elif change == "reset_adam":
        config["retention"]["groups"][1]["reset_adam"] = True
    else:
        path = tmp_path / "missing_adam.pt"
        source = _payload(config["retention"]["checkpoint"])
        source["actor_optimizer"]["state"] = {}
        torch.save(source, path)
        config["retention"].update(checkpoint=str(path), expected_sha256=_sha(path))
    config["output_dir"] = str(tmp_path / "must_not_exist")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        run_experiment(path)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("rate", [0.0, -1e-4, float("nan"), float("inf"), 3e-4])
def test_invalid_or_non_decreasing_actor_rates_are_rejected(tmp_path, lr_config, rate):
    from scripts.train_actor_lr_retention import run_experiment

    config = deepcopy(lr_config)
    config["retention"]["groups"][1]["actor_learning_rate"] = rate
    config["output_dir"] = str(tmp_path / "must_not_exist")
    path = tmp_path / "invalid_lr.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        run_experiment(path)
    assert not Path(config["output_dir"]).exists()


def test_formal_three_arm_configuration_preserves_the_v08_baseline():
    config = _yaml(ROOT / "configs/mappo_actor_lr_retention.yaml")
    original = _yaml(ROOT / "configs/mappo_capture500.yaml")
    for key in ("env_config", "seed", "device", "torch_num_threads", "network", "ppo", "normalization"):
        assert config[key] == original[key]
    assert config["training"]["total_env_steps"] == 20000
    assert config["training"]["rollout_steps"] == 512
    assert config["ppo"]["critic_learning_rate"] == 1e-3
    assert config["retention"]["evaluation_interval_steps"] == 2000
    assert config["retention"]["groups"] == [dict(name=name, reset_adam=False, actor_learning_rate=rate)
        for name, rate in zip(("A", "B", "C"), RATES)]
    assert "v08_" in config["retention"]["checkpoint"]
    assert config["retention"]["checkpoint"].endswith("checkpoints/update_000180.pt")
    assert not config.get("curriculum")
