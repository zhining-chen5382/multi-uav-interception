"""Exact reproduction checks using tiny real PPO runs, not performance claims."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from envs import PursuitEnv
from scripts import train_mappo
from scripts.verify_zero_direction_reproduction import (
    FixedZeroDirectionEnv, compare_csv, compare_run, first_difference,
    fixed_zero_training_environment, run_verification, validate_reference,
)
from utils.experiment import ROOT


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write(path, config):
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def tiny_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mappo, "run_post_training_analysis", lambda *args: {"status": "disabled"})
    env = _yaml(ROOT / "configs/baseline_capture500.yaml")
    env["environment"]["max_episode_steps"] = 3
    env_path = _write(tmp_path / "environment.yaml", env)
    config = _yaml(ROOT / "configs/mappo_capture500.yaml")
    config["env_config"] = str(env_path)
    config["output_dir"] = str(tmp_path / "reference")
    config["training"].update(total_env_steps=12, rollout_steps=4,
                               checkpoint_interval_updates=1, capture_rate_window=3)
    config["network"]["actor_hidden_sizes"] = [8]
    config["network"]["critic_hidden_sizes"] = [8]
    config["ppo"].update(update_epochs=1, minibatch_size=2)
    ref_path = _write(tmp_path / "reference.yaml", config)
    train_mappo.train(ref_path)
    reference = tmp_path / "reference"
    actual_config = deepcopy(config)
    actual_config["output_dir"] = str(tmp_path / "actual")
    actual_config["training"]["total_env_steps"] = 10
    actual_config["reproduction"] = dict(reference_run=str(reference), checkpoint_steps=[4, 8])
    actual_path = _write(tmp_path / "actual.yaml", actual_config)
    return dict(config=actual_config, config_path=actual_path,
                environment=env_path, reference=reference, actual=tmp_path / "actual")


def test_fixed_zero_uses_v09_environment_without_changing_physical_states():
    path = ROOT / "configs/baseline_capture500.yaml"
    ordinary, zero = PursuitEnv(path), FixedZeroDirectionEnv(path)
    state = random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    expected, expected_info = ordinary.reset(seed=17)
    actual, info = zero.reset(seed=17)
    assert zero.half_angle_degrees == 0.0 and zero.current_angle_degrees == 0.0
    sampling = deepcopy(zero.angle_rng.bit_generator.state)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(info["global_state"], expected_info["global_state"])
    for action in (np.zeros((3, 3)), np.ones((3, 3)), -np.ones((3, 3))):
        left, right = ordinary.step(action), zero.step(action)
        for index in (0, 1):
            np.testing.assert_array_equal(left[index], right[index])
        np.testing.assert_array_equal(ordinary.states, zero.states)
        np.testing.assert_array_equal(ordinary.accelerations, zero.accelerations)
    for _ in range(3):
        zero.reset()
    assert zero.angle_rng.bit_generator.state == sampling
    assert first_difference(random.getstate(), state[0]) is None
    assert first_difference(np.random.get_state(), state[1]) is None
    assert torch.equal(torch.get_rng_state(), state[2])


def test_trainer_environment_binding_restored_after_exception():
    original = train_mappo.PursuitEnv
    with pytest.raises(RuntimeError, match="intentional"):
        with fixed_zero_training_environment():
            assert train_mappo.PursuitEnv is FixedZeroDirectionEnv
            raise RuntimeError("intentional")
    assert train_mappo.PursuitEnv is original


def test_actual_full_rollouts_match_reference_models_optimizers_rng_and_metrics(tiny_runs):
    original = train_mappo.PursuitEnv
    result = run_verification(tiny_runs["config_path"])
    assert train_mappo.PursuitEnv is original
    assert result["all_comparable_state_equal"]
    assert result["actual_steps"] == 10
    assert result["compared_full_rollout_prefix_steps"] == 8
    assert result["excluded_final_short_rollout_steps"] == 2
    assert [row["env_steps"] for row in result["checkpoint_comparisons"]] == [4, 8]
    assert all(row["status"] == "equal" for row in result["requested_checkpoints"])
    assert all(row["all_equal"] for row in result["csv_comparisons"])
    saved = json.loads((tiny_runs["actual"] / "reproduction_comparison.json").read_text(encoding="utf-8"))
    assert saved == result
    # Compare-only repeats the read-only comparison without running training.
    assert run_verification(tiny_runs["config_path"], compare_only=True)["all_comparable_state_equal"]


def test_checkpoint_comparison_detects_actual_weight_mutation(tiny_runs):
    run_verification(tiny_runs["config_path"])
    path = tiny_runs["actual"] / "checkpoints/update_000002.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = next(key for key, value in payload["actor"].items() if value.is_floating_point())
    payload["actor"][key].view(-1)[0] += 1.0
    torch.save(payload, path)
    result = compare_run(tiny_runs["actual"], tiny_runs["reference"], [4, 8])
    assert not result["all_comparable_state_equal"]
    assert result["checkpoint_comparisons"][-1]["differences"]["actor"].endswith("tensor values")
    with pytest.raises(RuntimeError, match="Reproduction diverged"):
        run_verification(tiny_runs["config_path"], compare_only=True)


def test_csv_comparison_detects_metric_change_and_excludes_short_tail(tmp_path):
    fields = ("update", "env_steps", "loss")
    paths = [tmp_path / "actual.csv", tmp_path / "reference.csv"]
    for path, rows in zip(paths, [((1, 4, 1.0), (2, 6, 9.0)), ((1, 4, 1.0), (2, 8, 2.0))]):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
    result = compare_csv(paths[0], paths[1], 4)
    assert result["all_equal"] and result["rows_compared"] == 1
    text = paths[0].read_text(encoding="utf-8").replace("1.0", "1.1")
    paths[0].write_text(text, encoding="utf-8")
    result = compare_csv(paths[0], paths[1], 4)
    assert not result["all_equal"]
    assert result["first_difference"]["differences"]["loss"] == dict(actual="1.1", expected="1.0")


@pytest.mark.parametrize("key", ["seed", "ppo", "network", "normalization", "rollout_steps"])
def test_changed_learning_parameters_are_rejected_before_training(tiny_runs, monkeypatch, key):
    config = deepcopy(tiny_runs["config"])
    if key == "seed":
        config[key] += 1
    elif key == "rollout_steps":
        config["training"][key] += 1
    else:
        config[key]["unapproved_setting"] = 1
    _write(tiny_runs["config_path"], config)
    monkeypatch.setattr(train_mappo, "train", lambda *args, **kwargs: pytest.fail("Training must not start"))
    with pytest.raises(ValueError, match="differs from reference"):
        run_verification(tiny_runs["config_path"])
    assert not tiny_runs["actual"].exists()


def test_changed_environment_is_rejected(tiny_runs):
    environment = _yaml(tiny_runs["environment"])
    environment["capture"]["radius"] = 6.0
    _write(tiny_runs["environment"], environment)
    with pytest.raises(ValueError, match="Environment differs"):
        validate_reference(tiny_runs["config"], tiny_runs["environment"])


def test_historical_reference_cannot_be_used_as_output(tiny_runs):
    with pytest.raises(ValueError, match="independent"):
        run_verification(tiny_runs["config_path"], output_dir=tiny_runs["reference"])
    with pytest.raises(ValueError, match="historical"):
        compare_run(tiny_runs["reference"], tiny_runs["reference"], 4)


def test_requested_checkpoint_outside_short_check_is_reported_not_fabricated(tiny_runs):
    config = deepcopy(tiny_runs["config"])
    config["reproduction"]["checkpoint_steps"] = [4, 8, 12]
    _write(tiny_runs["config_path"], config)
    result = run_verification(tiny_runs["config_path"])
    assert result["all_comparable_state_equal"]
    assert result["requested_checkpoints"][-1] == dict(
        step=12, status="outside_completed_full_rollout_prefix", checkpoint=None)


def test_comparison_requires_requested_checkpoint_inside_completed_prefix(tiny_runs):
    run_verification(tiny_runs["config_path"])
    result = compare_run(tiny_runs["actual"], tiny_runs["reference"], [4, 7, 8])
    assert not result["all_comparable_state_equal"]
    assert result["requested_checkpoints"][1]["status"] == "missing"
