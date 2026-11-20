"""Short real-PPO acceptance for extending a completed run without restarting LR."""

from copy import deepcopy
import csv
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from algorithms.ppo import PPOAgent
from utils import formal_ppo_training as training


ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate(agent, environment, angles, seed, output):
    # A fixed score isolates best-checkpoint inheritance. Consuming all global
    # RNG streams also exercises the trainer's evaluation isolation.
    random.random()
    np.random.random(3)
    torch.rand(4)
    return dict(capture_rate=.8, defense_failure_rate=.2, timeout_rate=0.,
                average_interception_time=17., average_min_distance=4., average_return=100.)


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def same(left, right):
    assert training._state_hash(left) == training._state_hash(right)


def same_completed_training(left, right):
    first, second = deepcopy(left["agent"]), deepcopy(right["agent"])
    for saved in (first, second):
        saved["config"].pop("output_dir")
    same(first, second)
    same(left["training_state"], right["training_state"])
    same(left["rng_state"], right["rng_state"])


@pytest.fixture(scope="module")
def completed_source(tmp_path_factory):
    config = training.load_configuration(ROOT / "configs/ppo_formal_v1.yaml")
    config["environment_config"]["environment"]["max_episode_steps"] = 7
    config["training"].update(total_env_steps=24, num_envs=2, rollout_steps=4,
                              evaluation_interval_steps=6, validation_episodes=3)
    config["network"].update(actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8])
    config["ppo"].update(update_epochs=1, minibatch_size=4, target_kl=1000.)
    directory = tmp_path_factory.mktemp("budget_source") / "source_24"
    training.train(config, directory, 1001, evaluation_fn=evaluate)
    return dict(config=config, directory=directory, hashes=hashes(directory))


def continuation_config(source):
    config = deepcopy(source["config"])
    config["training"].update(total_env_steps=44, learning_rate_decay_steps=24)
    config["experiment"]["output_root"] = "outputs/formal_ppo_v1/test_continuation"
    return config


def test_scheduler_retains_legacy_bounds_and_requires_explicit_larger_budget(completed_source):
    legacy = PPOAgent(completed_source["config"])
    final_rates = (legacy.ppo["actor_final_learning_rate"], legacy.ppo["critic_final_learning_rate"])
    assert legacy.scheduler.step(24) == final_rates
    state = legacy.scheduler.state_dict()
    assert "maximum_env_steps" not in state
    with pytest.raises(ValueError, match="within budget"):
        legacy.scheduler.step(25)
    extended = PPOAgent(continuation_config(completed_source))
    with pytest.raises(ValueError, match="maximum_env_steps"):
        extended.scheduler.load_state_dict(state)
    extended.scheduler.load_state_dict(state, allow_budget_extension=True)
    assert extended.scheduler.environment_steps == 24
    assert extended.scheduler.total_env_steps == 24
    assert extended.scheduler.maximum_env_steps == 44
    assert extended.scheduler.step(32) == extended.scheduler.step(44) == final_rates
    assert extended.scheduler.state_dict()["maximum_env_steps"] == 44
    with pytest.raises(ValueError, match="within budget"):
        extended.scheduler.step(45)
    with pytest.raises(ValueError, match="monotonic"):
        extended.scheduler.step(43)


def test_completed_budget_extension_restores_all_state_and_preserves_prefix(completed_source, tmp_path):
    config = continuation_config(completed_source)
    source = completed_source["directory"]
    output = tmp_path / "extended"
    summary = training.train(config, output, 1001, resume_checkpoint=source / "checkpoints/final.pt",
                             allow_budget_extension=True, evaluation_fn=evaluate)
    before = checkpoint(source / "checkpoints/final.pt")
    resumed = checkpoint(output / "checkpoints/resume_initial.pt")
    same(before["training_state"], resumed["training_state"])
    same(before["rng_state"], resumed["rng_state"])
    for key in ("actor", "critic", "actor_optimizer", "critic_optimizer", "update_count"):
        same(before["agent"][key], resumed["agent"][key])
    assert resumed["agent"]["scheduler"]["total_env_steps"] == 24
    assert resumed["agent"]["scheduler"]["maximum_env_steps"] == 44
    assert summary["status"] == "completed" and summary["env_steps"] == 44
    assert summary["updates"] == 6
    assert [int(row["env_steps"]) for row in rows(output / "validation.csv")] == [6, 12, 18, 24, 30, 36, 42, 44]
    for name in training.LOG_FIELDS:
        prefix = rows(source / (name + ".csv"))
        assert rows(output / (name + ".csv"))[:len(prefix)] == prefix
    for row in rows(output / "updates.csv"):
        if int(row["env_steps"]) > 24:
            assert float(row["actor_learning_rate"]) == config["ppo"]["actor_final_learning_rate"]
            assert float(row["critic_learning_rate"]) == config["ppo"]["critic_final_learning_rate"]
    final = checkpoint(output / "checkpoints/final.pt")
    assert final["agent"]["scheduler"]["environment_steps"] == 44
    assert final["training_state"]["rollout_buffer"]["size"] == 0
    assert final["training_state"]["best_env_steps"] == before["training_state"]["best_env_steps"]
    same(checkpoint(output / "checkpoints/best.pt"), checkpoint(source / "checkpoints/best.pt"))
    assert hashes(source) == completed_source["hashes"]


def test_extended_run_partial_buffer_resume_is_bitwise(completed_source, tmp_path):
    config = continuation_config(completed_source)
    source = completed_source["directory"] / "checkpoints/final.pt"
    expected, paused, resumed = (tmp_path / name for name in ("expected", "paused", "resumed"))
    training.train(config, expected, 1001, resume_checkpoint=source,
                   allow_budget_extension=True, evaluation_fn=evaluate)
    paused_summary = training.train(config, paused, 1001, resume_checkpoint=source, stop_at_steps=34,
                                    allow_budget_extension=True, evaluation_fn=evaluate)
    assert paused_summary["pending_rollout_samples"] == 2
    partial = paused / "checkpoints/paused_000034.pt"
    partial_bytes = partial.read_bytes()
    training.train(config, resumed, 1001, resume_checkpoint=partial, evaluation_fn=evaluate)
    same_completed_training(checkpoint(expected / "checkpoints/final.pt"), checkpoint(resumed / "checkpoints/final.pt"))
    for name in training.LOG_FIELDS:
        assert rows(expected / (name + ".csv")) == rows(resumed / (name + ".csv"))
    assert partial.read_bytes() == partial_bytes
    assert hashes(completed_source["directory"]) == completed_source["hashes"]


@pytest.mark.parametrize("change", [
    "flag-missing", "missing-horizon", "changed-horizon", "same-budget", "smaller-budget",
    "actor-lr", "critic-lr", "entropy", "network", "environment", "seed", "validation-seed", "partial-source",
])
def test_extension_rejects_research_changes_and_incomplete_sources(completed_source, tmp_path, change):
    config = continuation_config(completed_source)
    source = completed_source["directory"] / "checkpoints/final.pt"
    seed, explicit = 1001, True
    if change == "flag-missing":
        explicit = False
    elif change == "missing-horizon":
        config["training"].pop("learning_rate_decay_steps")
    elif change == "changed-horizon":
        config["training"]["learning_rate_decay_steps"] = 26
    elif change == "same-budget":
        config["training"]["total_env_steps"] = 24
    elif change == "smaller-budget":
        config["training"].update(total_env_steps=22, learning_rate_decay_steps=20)
    elif change == "actor-lr":
        config["ppo"]["actor_learning_rate"] = .0001
    elif change == "critic-lr":
        config["ppo"]["critic_learning_rate"] = .0005
    elif change == "entropy":
        config["ppo"]["entropy_coefficient"] = .02
    elif change == "network":
        config["network"]["actor_hidden_sizes"] = [9, 9]
    elif change == "environment":
        config["environment_config"]["capture"]["radius"] = 6.
    elif change == "seed":
        seed = 1002
    elif change == "validation-seed":
        config["experiment"]["validation_seed"] += 1
    else:
        source = completed_source["directory"] / "checkpoints/evaluated_000006.pt"
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        training.train(config, output, seed, resume_checkpoint=source,
                       allow_budget_extension=explicit, evaluation_fn=evaluate)
    assert not output.exists()
    assert hashes(completed_source["directory"]) == completed_source["hashes"]


def test_extension_requires_checkpoint_and_preserves_existing_directory(completed_source, tmp_path):
    config = continuation_config(completed_source)
    absent = tmp_path / "absent"
    with pytest.raises(ValueError, match="source checkpoint"):
        training.train(config, absent, 1001, allow_budget_extension=True, evaluation_fn=evaluate)
    assert not absent.exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="new output directory"):
        training.train(config, existing, 1001,
                       resume_checkpoint=completed_source["directory"] / "checkpoints/final.pt",
                       allow_budget_extension=True, evaluation_fn=evaluate)
    assert marker.read_bytes() == b"preserve"


@pytest.mark.parametrize("change", ["negative", "too-long", "noninteger"])
def test_decay_horizon_configuration_is_validated(completed_source, tmp_path, change):
    config = continuation_config(completed_source)
    config["training"]["learning_rate_decay_steps"] = {"negative": -1, "too-long": 46, "noninteger": 24.5}[change]
    with pytest.raises(ValueError):
        training.train(config, tmp_path / "invalid", 1001)
    assert not (tmp_path / "invalid").exists()
