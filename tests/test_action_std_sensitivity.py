"""Frozen-policy sampling isolation; tiny fixtures never perform PPO training."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.mappo import MAPPOAgent
from envs import PursuitEnv
from scripts.evaluate_action_std_sensitivity import (
    FrozenStdSampler, distribution_summary, evaluate, representatives,
    validate_protocol, wilson_interval,
)
from utils.evaluation import run_episode, save_trajectory


ROOT = Path(__file__).resolve().parents[1]
MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 1.5)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def frozen_agent():
    config = yaml.safe_load((ROOT / "configs/mappo_capture500.yaml").read_text(encoding="utf-8"))
    config["network"]["actor_hidden_sizes"] = [8]
    config["network"]["critic_hidden_sizes"] = [8]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(413)
        agent = MAPPOAgent(config, device="cpu")
    for model in (agent.actor, agent.critic):
        model.eval()
        model.requires_grad_(False)
    # Nonempty optimizer-memory fixtures are assigned directly, without an
    # optimizer step, to catch accidental changes to retained Adam moments.
    for optimizer in (agent.actor_optimizer, agent.critic_optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                optimizer.state[parameter] = {
                    "step": torch.tensor(17.0),
                    "exp_avg": torch.full_like(parameter, 0.01),
                    "exp_avg_sq": torch.full_like(parameter, 0.02),
                }
    return agent


@pytest.fixture
def observations():
    return np.linspace(-2.0, 3.0, 63, dtype=np.float32).reshape(3, 21)


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def test_zero_multiplier_exactly_matches_native_deterministic_actions(frozen_agent, observations):
    sampler = FrozenStdSampler(frozen_agent, 0.0, 1000)
    expected = frozen_agent.act(observations, deterministic=True)[0]
    before = torch.get_rng_state().clone()
    first = sampler.act(observations)
    assert first[1:] == (None, None, None)
    np.testing.assert_array_equal(first[0], expected)
    np.testing.assert_array_equal(sampler.act(observations)[0], expected)
    np.testing.assert_array_equal(sampler.history["raw_actions"][0], sampler.history["means"][0])
    assert np.count_nonzero(sampler.history["noise"]) == 0
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("multiplier", MULTIPLIERS)
def test_sampling_scales_latent_std_before_tanh(frozen_agent, observations, multiplier):
    sampler = FrozenStdSampler(frozen_agent, multiplier, 1029)
    actions = sampler.act(observations)[0]
    history = {key: values[0] for key, values in sampler.history.items()}
    assert set(history) == {"raw_actions", "means", "noise", "base_stds"}
    assert all(array.shape == (3, 3) and np.isfinite(array).all() for array in history.values())
    expected_raw = history["means"] + multiplier * history["base_stds"] * history["noise"]
    np.testing.assert_array_equal(history["raw_actions"], expected_raw)
    np.testing.assert_allclose(actions, np.tanh(expected_raw), atol=1e-7, rtol=1e-6)
    assert actions.shape == (3, 3) and np.isfinite(actions).all()
    assert np.all(actions >= -1) and np.all(actions <= 1)


def test_same_seed_pairs_standard_normal_prefixes_across_multipliers(frozen_agent, observations):
    samplers = [FrozenStdSampler(frozen_agent, multiplier, 1047) for multiplier in MULTIPLIERS[1:]]
    for step in range(4):
        for sampler in samplers:
            sampler.act(observations + step * 0.1)
        for sampler in samplers[1:]:
            np.testing.assert_array_equal(sampler.history["noise"][step], samplers[0].history["noise"][step])
            np.testing.assert_array_equal(sampler.history["means"][step], samplers[0].history["means"][step])
    assert not np.array_equal(samplers[0].history["noise"][0], samplers[0].history["noise"][1])


def test_one_multiplier_reproduces_native_stochastic_sampling_with_same_seed(frozen_agent, observations):
    sampler = FrozenStdSampler(frozen_agent, 1.0, 1091)
    actual = [sampler.act(observations + step * 0.1)[0] for step in range(3)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1091)
        expected = [frozen_agent.act(observations + step * 0.1, deterministic=False)[0]
                    for step in range(3)]
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_reproducible_seeds_without_changing_global_torch_rng(frozen_agent, observations):
    before = torch.get_rng_state().clone()
    first = FrozenStdSampler(frozen_agent, 0.5, 1000).act(observations)[0]
    same = FrozenStdSampler(frozen_agent, 0.5, 1000).act(observations)[0]
    different = FrozenStdSampler(frozen_agent, 0.5, 1001).act(observations)[0]
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, different)
    assert torch.equal(torch.get_rng_state(), before)


def test_sampling_preserves_all_networks_adam_memory_modes_and_gradients(frozen_agent, observations, monkeypatch):
    before = {
        "actor": deepcopy(frozen_agent.actor.state_dict()),
        "critic": deepcopy(frozen_agent.critic.state_dict()),
        "actor_optimizer": deepcopy(frozen_agent.actor_optimizer.state_dict()),
        "critic_optimizer": deepcopy(frozen_agent.critic_optimizer.state_dict()),
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen evaluation must not update or execute the centralized critic")

    monkeypatch.setattr(frozen_agent, "update", forbidden)
    monkeypatch.setattr(frozen_agent.actor_optimizer, "step", forbidden)
    monkeypatch.setattr(frozen_agent.critic_optimizer, "step", forbidden)
    monkeypatch.setattr(frozen_agent.critic, "forward", forbidden)
    for multiplier in MULTIPLIERS:
        sampler = FrozenStdSampler(frozen_agent, multiplier, 1000)
        sampler.act(observations, global_state=np.zeros(24))
        sampler.sampling_report()
    for key, expected in before.items():
        _assert_equal(getattr(frozen_agent, key).state_dict(), expected)
    for model in (frozen_agent.actor, frozen_agent.critic):
        assert not model.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())


def test_report_and_sampling_use_clamped_log_std_without_modifying_parameter(frozen_agent, observations):
    values = torch.tensor([-9.0, -0.4, 9.0])
    with torch.no_grad():
        frozen_agent.actor.log_std.copy_(values)
    sampler = FrozenStdSampler(frozen_agent, 1.5, 1051)
    report = sampler.sampling_report()
    effective = values.clamp(frozen_agent.actor.min_log_std, frozen_agent.actor.max_log_std)
    np.testing.assert_array_equal(report["log_std"], values.numpy())
    np.testing.assert_array_equal(report["effective_log_std"], effective.numpy())
    np.testing.assert_array_equal(report["base_std"], effective.exp().numpy())
    np.testing.assert_array_equal(report["scaled_std"], (effective.exp() * 1.5).numpy())
    sampler.act(observations)
    np.testing.assert_array_equal(sampler.history["base_stds"][0], np.tile(effective.exp().numpy(), (3, 1)))
    assert torch.equal(frozen_agent.actor.log_std, values)


@pytest.mark.parametrize("bad", [-0.25, np.nan, np.inf, -np.inf, True, "0.5", None])
def test_invalid_sampler_multiplier_rejected(frozen_agent, bad):
    with pytest.raises(ValueError):
        FrozenStdSampler(frozen_agent, bad, 1000)


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_sampler_seed_rejected(frozen_agent, bad):
    with pytest.raises(ValueError):
        FrozenStdSampler(frozen_agent, 1.0, bad)


@pytest.mark.parametrize("bad", [np.zeros(21), np.zeros((3, 20)), np.full((3, 21), np.nan)])
def test_invalid_observation_rejected(frozen_agent, bad):
    with pytest.raises(ValueError):
        FrozenStdSampler(frozen_agent, 1.0, 1000).act(bad)


@pytest.fixture
def protocol():
    return yaml.safe_load((ROOT / "configs/eval_action_std_sensitivity.yaml").read_text(encoding="utf-8"))


def test_real_protocol_contains_only_five_prescribed_arms_with_paired_seeds(protocol):
    validate_protocol(protocol)
    assert protocol["episodes_per_group"] == 100
    assert [group["multiplier"] for group in protocol["groups"]] == list(MULTIPLIERS)
    assert protocol["expected_env_steps"] == 92160
    assert protocol["base_seed"] == 1000
    assert "learning_rate" not in protocol and "entropy" not in protocol


@pytest.mark.parametrize("key,bad", [
    ("episodes_per_group", 0), ("episodes_per_group", True),
    ("episodes_per_group", 1.5), ("base_seed", -1),
    ("base_seed", True), ("base_seed", 2 ** 63),
    ("groups", []), ("groups", {}),
    ("representative_selection", "highest_return"),
])
def test_invalid_protocol_values_rejected(protocol, key, bad):
    protocol[key] = bad
    with pytest.raises(ValueError):
        validate_protocol(protocol)


@pytest.mark.parametrize("kind", ["duplicate_name", "duplicate_multiplier", "unsafe_name", "nan_multiplier"])
def test_ambiguous_or_unsafe_groups_rejected(protocol, kind):
    if kind == "duplicate_name":
        protocol["groups"][1]["name"] = protocol["groups"][0]["name"]
    elif kind == "duplicate_multiplier":
        protocol["groups"][1]["multiplier"] = protocol["groups"][0]["multiplier"]
    elif kind == "unsafe_name":
        protocol["groups"][1]["name"] = "../overwrite"
    else:
        protocol["groups"][1]["multiplier"] = float("nan")
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_distribution_summary_uses_actual_values_and_preserves_zero_variance():
    actual = distribution_summary([1.0, 3.0, 5.0, 7.0])
    assert actual["mean"] == actual["median"] == 4
    assert actual["min"] == 1 and actual["max"] == 7
    assert actual["p25"] == 2.5 and actual["p75"] == 5.5
    assert actual["std"] == np.std([1.0, 3.0, 5.0, 7.0], ddof=0)
    assert distribution_summary([3.0] * 100)["std"] == 0
    assert 0 <= wilson_interval(0, 100)[0] <= wilson_interval(0, 100)[1] < 0.04
    assert 0.96 < wilson_interval(100, 100)[0] <= wilson_interval(100, 100)[1] <= 1


def test_representative_is_median_not_best_success_and_failure_are_explicit():
    rows = [dict(episode_index=i, seed=1000+i, min_distance=distance,
                 captured=distance < 5, trajectory="episode_{}.npz".format(i))
            for i, distance in enumerate([2.0, 3.0, 20.0, 30.0, 40.0])]
    actual = representatives(rows)
    assert actual["typical"]["episode_index"] == 2
    assert actual["success"]["episode_index"] == 0
    assert actual["failure"]["episode_index"] == 3
    assert "ALL" in actual["typical"]["selection"]
    assert "success" in actual["success"]["selection"]


def test_tiny_frozen_evaluation_preserves_sources_and_saves_replayable_noise(tmp_path, frozen_agent, monkeypatch):
    # This is a three-step fixture rollout, not a model-performance experiment.
    from scripts import evaluate_action_std_sensitivity as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    environment = yaml.safe_load((ROOT / "configs/baseline_capture500.yaml").read_text(encoding="utf-8"))
    environment["environment"]["max_episode_steps"] = 3
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text(yaml.safe_dump(environment), encoding="utf-8")
    checkpoint = tmp_path / "source_fixture.pt"
    frozen_agent.save(checkpoint, training_state={"env_steps": 0, "update": 0, "episodes": 0},
                      metadata={"environment_config": environment})
    metrics, trajectory = run_episode(PursuitEnv(environment_path), frozen_agent, seed=1000)
    reference = tmp_path / "reference.npz"
    save_trajectory(reference, trajectory, metrics, seed=1000)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_sha, reference_sha = sha(checkpoint), sha(reference)
    config = dict(checkpoint=str(checkpoint), expected_checkpoint_sha256=source_sha,
                  expected_env_steps=0, reference_environment_config=str(environment_path),
                  reference_trajectory=str(reference), output_dir=str(tmp_path / "evaluation"),
                  episodes_per_group=2, base_seed=1000,
                  groups=[dict(name="std_0", multiplier=0.0), dict(name="std_1", multiplier=1.0)])
    config_path = tmp_path / "protocol.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("No training, optimizer steps, or new checkpoint saving permitted")

    monkeypatch.setattr(MAPPOAgent, "update", forbidden)
    monkeypatch.setattr(MAPPOAgent, "save", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    before = torch.get_rng_state().clone()
    result = evaluate(config_path)
    assert result["episodes_total"] == 4 and result["caller_global_torch_rng_restored"]
    assert result["frozen_state_verification"]["all_checks_passed"]
    assert torch.equal(torch.get_rng_state(), before)
    assert sha(checkpoint) == source_sha and sha(reference) == reference_sha
    output = Path(config["output_dir"])
    with (output / "episode_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert [int(row["seed"]) for row in rows] == [1000, 1001, 1000, 1001]
    for row in rows:
        with np.load(output / row["trajectory"], allow_pickle=False) as archive:
            saved = json.loads(archive["metadata_json"].item())
            assert saved["seed"] == int(row["seed"])
            assert archive["states"].shape == (4, 4, 6)
            raw = archive["latent_raw_actions"]
            mean = archive["latent_means"]
            std = archive["latent_base_stds"]
            noise = archive["latent_noise"]
            assert raw.shape == mean.shape == std.shape == noise.shape == (3, 3, 3)
            np.testing.assert_array_equal(raw, mean + float(row["multiplier"]) * std * noise)
            np.testing.assert_allclose(archive["actions"], np.tanh(raw), rtol=1e-6, atol=1e-7)
    prior_metrics = (output / "episode_metrics.csv").read_bytes()
    with pytest.raises(FileExistsError):
        evaluate(config_path)
    assert (output / "episode_metrics.csv").read_bytes() == prior_metrics
