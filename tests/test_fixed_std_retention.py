"""Actual short PPO checks for fixed-std retention, not task-performance tests."""

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from algorithms.mappo import MAPPOAgent
from envs import PursuitEnv


ROOT = Path(__file__).resolve().parents[1]
MULTIPLIERS = (1.0, 0.15, 0.10)


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
def fixed_config(tmp_path, monkeypatch):
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
    source_config = tmp_path / "source_config.yaml"
    source_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # Populate actual, nonzero Adam moments/counters in a tiny isolated source.
    train_mappo.train(source_config)
    checkpoint = tmp_path / "source/checkpoints/final.pt"
    source = _payload(checkpoint)
    assert source["actor_optimizer"]["state"] and source["critic_optimizer"]["state"]
    config["training"]["total_env_steps"] = 13
    config["output_dir"] = str(tmp_path / "fixed_std_triplet")
    config["retention"] = dict(checkpoint=str(checkpoint), expected_sha256=_sha(checkpoint),
        evaluation_interval_steps=4, evaluation_seed=1000,
        groups=[dict(name=name, reset_adam=False, frozen_std_multiplier=multiplier)
                for name, multiplier in zip(("A", "B", "C"), MULTIPLIERS)])
    return config


def _context(config):
    from scripts.train_fixed_std_retention import validate_config

    return validate_config(config, _yaml(config["env_config"]))


@pytest.fixture
def trained_triplet(tmp_path, fixed_config):
    from scripts.train_fixed_std_retention import run_experiment

    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(fixed_config, sort_keys=False), encoding="utf-8")
    run_experiment(path)
    return Path(fixed_config["output_dir"])


@pytest.mark.parametrize("multiplier", MULTIPLIERS)
def test_initialization_scales_only_log_std_without_resetting_any_adam_state(fixed_config, multiplier):
    from scripts.train_policy_retention import initialize_agent

    context = _context(fixed_config)
    source = _payload(fixed_config["retention"]["checkpoint"])
    agent, _ = initialize_agent(fixed_config, context, False, frozen_std_multiplier=multiplier)
    expected_actor = deepcopy(source["actor"])
    expected_actor["log_std"] = expected_actor["log_std"] + math.log(multiplier)
    _equal(agent.actor.state_dict(), expected_actor)
    _equal(agent.critic.state_dict(), source["critic"])
    _equal(agent.actor_optimizer.state_dict(), source["actor_optimizer"])
    _equal(agent.critic_optimizer.state_dict(), source["critic_optimizer"])
    assert agent.actor.log_std.requires_grad is False
    assert agent.actor.log_std.grad is None
    assert all(p.requires_grad for p in agent.actor.mean.parameters())
    assert all(p.requires_grad for p in agent.critic.parameters())
    assert any(p is agent.actor.log_std for group in agent.actor_optimizer.param_groups for p in group["params"])
    assert agent.actor_optimizer.state[agent.actor.log_std]
    assert agent.actor_optimizer.param_groups[0]["lr"] == 3e-4
    assert agent.critic_optimizer.param_groups[0]["lr"] == 1e-3
    _equal(random.getstate(), source["rng_state"]["python"])
    _equal(np.random.get_state(), source["rng_state"]["numpy"])
    _equal(torch.get_rng_state(), source["rng_state"]["torch"])
    assert _sha(Path(fixed_config["retention"]["checkpoint"])) == fixed_config["retention"]["expected_sha256"]


@pytest.mark.parametrize("multiplier", MULTIPLIERS)
def test_sampling_log_prob_and_ppo_ratio_use_the_same_fixed_distribution(fixed_config, multiplier):
    from scripts.train_policy_retention import initialize_agent

    source = _context(fixed_config)
    agent, payload = initialize_agent(fixed_config, source, False, frozen_std_multiplier=multiplier)
    env = PursuitEnv(fixed_config["env_config"])
    observations, info = env.reset(seed=fixed_config["seed"])
    actions, log_prob, raw, _ = agent.act(observations, info["global_state"])
    obs = agent._tensor(observations)
    raw_tensor = agent._tensor(raw)
    with torch.no_grad():
        distribution = agent.actor.distribution(obs)
        recalculated, entropy = agent.actor.evaluate_actions(obs, raw_tensor)
        source_std = payload["actor"]["log_std"].clamp(
            agent.actor.min_log_std, agent.actor.max_log_std).exp()
    np.testing.assert_array_equal(recalculated.numpy(), log_prob)
    np.testing.assert_array_equal((recalculated - agent._tensor(log_prob)).exp().numpy(), np.ones_like(log_prob))
    np.testing.assert_allclose(distribution.scale.numpy(),
                               np.broadcast_to((source_std * multiplier).numpy(), (3, 3)), rtol=3e-7, atol=1e-8)
    np.testing.assert_allclose(actions, np.tanh(raw), rtol=1e-6, atol=1e-7)
    assert actions.shape == raw.shape == (3, 3)
    assert np.isfinite(actions).all() and np.isfinite(raw).all() and np.isfinite(entropy.numpy()).all()
    assert agent.actor.log_std.grad is None


def test_every_actual_ppo_update_keeps_std_and_its_adam_memory_but_trains_mean_and_critic(tmp_path, fixed_config, monkeypatch):
    from scripts.train_fixed_std_retention import run_experiment

    actual_update = MAPPOAgent.update
    updates = []

    def checked_update(agent, buffer):
        before_std = agent.actor.log_std.detach().clone()
        before_std_adam = deepcopy(agent.actor_optimizer.state[agent.actor.log_std])
        before_mean = deepcopy(agent.actor.mean.state_dict())
        before_critic = deepcopy(agent.critic.state_dict())
        assert not agent.actor.log_std.requires_grad and agent.actor.log_std.grad is None
        # Entropy estimation draws RNG; the observer must not perturb PPO RNG.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            observations = agent._tensor(buffer.observations[:buffer.size]).reshape(-1, agent.observation_dim)
            raw = agent._tensor(buffer.raw_actions[:buffer.size]).reshape(-1, agent.action_dim)
            log_probs, _ = agent.actor.evaluate_actions(observations, raw)
            old = agent._tensor(buffer.log_probs[:buffer.size]).reshape(-1)
            torch.testing.assert_close((log_probs - old).exp(), torch.ones_like(old), rtol=0, atol=1e-5)
        result = actual_update(agent, buffer)
        assert torch.equal(agent.actor.log_std, before_std)
        _equal(agent.actor_optimizer.state[agent.actor.log_std], before_std_adam)
        assert agent.actor.log_std.grad is None
        assert any(not torch.equal(value, before_mean[name]) for name, value in agent.actor.mean.state_dict().items())
        assert any(not torch.equal(value, before_critic[name]) for name, value in agent.critic.state_dict().items())
        assert all(np.isfinite(value) for value in result.values())
        updates.append(buffer.size)
        return result

    monkeypatch.setattr(MAPPOAgent, "update", checked_update)
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(fixed_config, sort_keys=False), encoding="utf-8")
    run_experiment(path)
    assert updates == [3, 3, 3, 3, 1] * 3


def test_initial_deterministic_policies_rng_and_adam_are_shared_but_stochastic_batches_differ(trained_triplet, fixed_config):
    source = _payload(fixed_config["retention"]["checkpoint"])
    initial_actions, initial_trajectories, stochastic_actions = [], [], []
    for name, multiplier in zip(("A", "B", "C"), MULTIPLIERS):
        directory = trained_triplet / name
        initial = _payload(directory / "checkpoints/initial.pt")
        expected = deepcopy(source["actor"])
        expected["log_std"] += math.log(multiplier)
        _equal(initial["actor"], expected)
        for key in ("critic", "actor_optimizer", "critic_optimizer", "rng_state"):
            _equal(initial[key], source[key])
        assert initial["training_state"]["env_steps"] == 0
        with np.load(directory / "evaluations/at_000000/first_episode.npz", allow_pickle=False) as archive:
            initial_actions.append(archive["actions"].copy())
            initial_trajectories.append(archive["states"].copy())
        with np.load(directory / "first_rollout.npz", allow_pickle=False) as archive:
            stochastic_actions.append(archive["raw_actions"].copy())
    for actions, states in zip(initial_actions[1:], initial_trajectories[1:]):
        np.testing.assert_array_equal(actions, initial_actions[0])
        np.testing.assert_array_equal(states, initial_trajectories[0])
    assert all(not np.array_equal(stochastic_actions[0], value) for value in stochastic_actions[1:])
    assert not np.array_equal(stochastic_actions[1], stochastic_actions[2])
    assert _sha(Path(fixed_config["retention"]["checkpoint"])) == fixed_config["retention"]["expected_sha256"]


def test_exact_evaluation_schedule_complete_rollouts_and_real_final_update(trained_triplet):
    for name in ("A", "B", "C"):
        directory = trained_triplet / name
        updates, evaluations = _rows(directory / "updates.csv"), _rows(directory / "evaluations.csv")
        assert [int(row["env_steps"]) for row in updates] == [3, 6, 9, 12, 13]
        assert [int(row["rollout_steps"]) for row in updates] == [3, 3, 3, 3, 1]
        assert [int(row["env_steps"]) for row in evaluations] == [0, 4, 8, 12, 13]
        assert [int(row["last_update_env_steps"]) for row in evaluations] == [0, 3, 6, 12, 13]
        assert [int(row["pending_rollout_steps"]) for row in evaluations] == [0, 1, 2, 0, 0]
        assert evaluations[-1]["checked_after_update"] == "True"
        for row in updates:
            assert float(row["actor_learning_rate"]) == 3e-4
            assert float(row["critic_learning_rate"]) == 1e-3
            for metric in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
                assert np.isfinite(float(row[metric]))
            assert 0 <= float(row["clip_fraction"]) <= 1
        final = _payload(directory / "checkpoints/final.pt")
        assert final["training_state"]["env_steps"] == final["training_state"]["last_update_env_steps"] == 13
        assert final["training_state"]["pending_rollout_steps"] == 0
        assert final["training_state"]["fully_resumable"] is False
        assert Path(directory / "checkpoints/best.pt").is_file()
        for row in evaluations:
            with np.load(row["trajectory"], allow_pickle=False) as archive:
                metrics = json.loads(archive["metadata_json"].item())["metrics"]
                assert archive["states"].shape == (metrics["episode_length"] + 1, 4, 6)
                assert archive["actions"].shape == (metrics["episode_length"], 3, 3)


def test_fixed_std_audit_rows_and_all_saved_checkpoints_preserve_the_scaled_std(trained_triplet, fixed_config):
    original = _payload(fixed_config["retention"]["checkpoint"])
    source_agent, _ = MAPPOAgent.from_checkpoint(fixed_config["retention"]["checkpoint"])
    source_serial = source_agent.actor_optimizer.state_dict()
    std_id = next(serial for group, saved in zip(source_agent.actor_optimizer.param_groups, source_serial["param_groups"])
                  for parameter, serial in zip(group["params"], saved["params"]) if parameter is source_agent.actor.log_std)
    for name, multiplier in zip(("A", "B", "C"), MULTIPLIERS):
        directory = trained_triplet / name
        expected_std = original["actor"]["log_std"] + math.log(multiplier)
        proof = json.loads((directory / "initialization_verification.json").read_text(encoding="utf-8"))
        assert proof["all_checks_passed"]
        assert proof["frozen_std"]["multiplier"] == multiplier
        assert proof["frozen_std"]["requires_grad"] is False
        assert proof["frozen_std"]["optimizer_parameter_and_state_retained"] is True
        rows = _rows(directory / "fixed_std_updates.csv")
        assert len(rows) == 5
        for row in rows:
            assert row["log_std_requires_grad"] == "False"
            for key in ("log_std_grad_is_none", "frozen_log_std_unchanged", "frozen_log_std_adam_unchanged"):
                assert row[key] == "True"
            np.testing.assert_allclose([float(row["log_std_" + axis]) for axis in "xyz"], expected_std.numpy(), rtol=0, atol=1e-7)
            np.testing.assert_allclose([float(row["std_" + axis]) for axis in "xyz"], expected_std.exp().numpy(), rtol=0, atol=1e-7)
        for checkpoint in (directory / "checkpoints").glob("*.pt"):
            payload = _payload(checkpoint)
            _equal(payload["actor"]["log_std"], expected_std)
            _equal(payload["actor_optimizer"]["state"][std_id], original["actor_optimizer"]["state"][std_id])
        reloaded, _ = MAPPOAgent.from_checkpoint(directory / "checkpoints/final.pt")
        _equal(reloaded.actor.log_std.detach(), expected_std)
        # requires_grad is a training setup flag, not serialized state_dict data.
        # Generic from_checkpoint is checked for inference values, not resume promises.
        assert reloaded.actor_optimizer.param_groups[0]["lr"] == 3e-4


def test_legacy_initializer_keeps_log_std_trainable_and_does_not_apply_an_implicit_scale(fixed_config):
    from scripts.train_policy_retention import initialize_agent

    source = _context(fixed_config)
    payload = _payload(fixed_config["retention"]["checkpoint"])
    agent, _ = initialize_agent(fixed_config, source, False)
    assert agent.actor.log_std.requires_grad is True
    _equal(agent.actor.state_dict(), payload["actor"])
    _equal(agent.actor_optimizer.state_dict(), payload["actor_optimizer"])


@pytest.mark.parametrize("multiplier", [0.0, -0.1, float("nan"), float("inf"), True, 1e-12, 1e12])
def test_invalid_fixed_std_is_rejected_without_writing_source(fixed_config, multiplier):
    from scripts.train_policy_retention import initialize_agent

    source = _context(fixed_config)
    with pytest.raises(ValueError):
        initialize_agent(fixed_config, source, False, frozen_std_multiplier=multiplier)
    assert _sha(Path(fixed_config["retention"]["checkpoint"])) == fixed_config["retention"]["expected_sha256"]


@pytest.mark.parametrize("change", ["hash", "reward", "actor_lr", "critic_lr", "entropy", "gamma", "network", "reset_adam", "missing_adam"])
def test_unapproved_changes_fail_before_any_output_is_created(tmp_path, fixed_config, change):
    from scripts.train_fixed_std_retention import run_experiment

    config = deepcopy(fixed_config)
    if change == "hash":
        config["retention"]["expected_sha256"] = "0" * 64
    elif change == "reward":
        environment = _yaml(config["env_config"])
        environment["reward"]["capture_reward"] += 1
        path = tmp_path / "changed_environment.yaml"
        path.write_text(yaml.safe_dump(environment), encoding="utf-8")
        config["env_config"] = str(path)
    elif change in ("actor_lr", "critic_lr", "entropy", "gamma"):
        field = {"actor_lr": "actor_learning_rate", "critic_lr": "critic_learning_rate",
                 "entropy": "entropy_coefficient", "gamma": "gamma"}[change]
        config["ppo"][field] *= 0.5
    elif change == "network":
        config["network"]["actor_hidden_sizes"] = [17, 17]
    elif change == "reset_adam":
        config["retention"]["groups"][1]["reset_adam"] = True
    else:
        path = tmp_path / "missing_adam.pt"
        payload = _payload(config["retention"]["checkpoint"])
        payload["actor_optimizer"]["state"] = {}
        torch.save(payload, path)
        config["retention"].update(checkpoint=str(path), expected_sha256=_sha(path))
    config["output_dir"] = str(tmp_path / "must_not_exist")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        run_experiment(path)
    assert not Path(config["output_dir"]).exists()


def test_formal_configuration_changes_only_the_fixed_std_treatment():
    config = _yaml(ROOT / "configs/mappo_fixed_std_retention.yaml")
    original = _yaml(ROOT / "configs/mappo_capture500.yaml")
    for key in ("env_config", "seed", "device", "torch_num_threads", "network", "ppo", "normalization"):
        assert config[key] == original[key]
    assert config["training"]["total_env_steps"] == 20000
    assert config["training"]["rollout_steps"] == 512
    assert config["retention"]["evaluation_interval_steps"] == 2000
    assert config["retention"]["groups"] == [dict(name=name, reset_adam=False, frozen_std_multiplier=value)
        for name, value in zip(("A", "B", "C"), MULTIPLIERS)]
    assert config["retention"]["checkpoint"].endswith("checkpoints/update_000180.pt")
    assert not config.get("curriculum") and not config.get("warm_start")
