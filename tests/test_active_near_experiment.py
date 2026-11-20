"""Controlled two-group protocol, source protection and orchestration tests."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import run_active_near_experiment as runner
from utils.formal_ppo_config import load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import create_scenarios


ROOT = Path(__file__).resolve().parents[1]


def configurations():
    return runner.load_configs()


def test_two_groups_only_change_the_confirmed_reward_settings():
    baseline = load_formal_config(ROOT / "configs/ppo_line_cost_k0p1.yaml")
    for beta, config in zip((100.0, 200.0), configurations()):
        before = deepcopy(config)
        assert validate_readiness(config, expected_seed_count=3) == config
        environment = deepcopy(config["environment_config"])
        shaping = environment["reward"]["best_so_far_shaping"]
        assert shaping.pop("mode") == "motion_attributed"
        assert shaping["beta"] == beta
        assert shaping["length_scale_m"] == 20.0
        assert environment["reward"]["line_cost_coefficient"] == 0.01
        shaping["beta"] = 100.0
        environment["reward"]["line_cost_coefficient"] = 0.1
        assert environment == baseline["environment_config"]
        for field in ("network", "ppo", "training", "device", "torch_num_threads", "acceptance"):
            assert config[field] == baseline[field]
        expected = deepcopy(baseline["experiment"])
        expected["output_root"] = config["experiment"]["output_root"]
        assert config["experiment"] == expected
        assert config == before
    assert configurations()[0]["training"]["total_env_steps"] == 300000
    assert configurations()[0]["training"]["num_envs"] == 8


def test_groups_share_all_validation_and_test_scenes(tmp_path):
    configs = configurations()
    manifests = [create_scenarios(config, config["environment_config"], tmp_path / str(index))
                 for index, config in enumerate(configs)]
    for field in ("validation_angles", "test_angles"):
        np.testing.assert_array_equal(manifests[0][field], manifests[1][field])
    assert manifests[0]["validation_sha256"] == manifests[1]["validation_sha256"]
    assert manifests[0]["test_sha256"] == manifests[1]["test_sha256"]


@pytest.fixture
def historical(tmp_path):
    config = load_formal_config(ROOT / "configs/ppo_line_cost_k0p1.yaml")
    root = tmp_path / "historical"
    create_scenarios(config, config["environment_config"], root / "scenarios")
    for seed in config["experiment"]["seeds"]:
        directory = root / ("seed_%d" % seed)
        (directory / "checkpoints").mkdir(parents=True)
        torch.save(dict(training_state={"env_steps": 300000}, config=config,
                        environment_config=config["environment_config"]), directory / "checkpoints/final.pt")
        for relative in ("summary.json", "environment_config.yaml", "training_config.yaml", "episodes.csv",
                         "updates.csv", "validation.csv", "checkpoints/initial.pt", "checkpoints/best.pt",
                         "final_test/episodes.csv", "final_model_test/episodes.csv"):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"protected historical fixture")
    return root


def test_preflight_matches_protocol_and_hashes_without_mutation(historical):
    for config in configurations():
        before = deepcopy(config)
        checked = runner.preflight(config, historical)
        assert checked["ready"] is True
        assert checked["steps_per_seed"] == 300000
        assert checked["seeds"] == [1001, 1002, 1003]
        assert len(checked["historical_source_sha256"]) == 36
        runner._verify_hashes(checked["historical_source_sha256"])
        assert config == before


@pytest.mark.parametrize("setting", ["capture", "initialization", "learning_rate", "evaluation"])
def test_preflight_rejects_unrequested_protocol_changes(historical, setting):
    config = configurations()[0]
    if setting == "capture":
        config["environment_config"]["capture"]["radius"] = 6.0
    elif setting == "initialization":
        config["environment_config"]["initialization"]["pursuer_position"][0] = 3.0
    elif setting == "learning_rate":
        config["ppo"]["critic_learning_rate"] = 0.002
    else:
        config["experiment"]["test_seed"] = 3002
    with pytest.raises(ValueError):
        runner.preflight(config, historical)


def test_group_preparation_snapshots_new_sources_and_checks_reused_scenes(tmp_path, historical, monkeypatch):
    config = configurations()[0]
    checked = runner.preflight(config, historical)

    def snapshot(output, actual_config):
        (output / "source_snapshot").mkdir()
        (output / "source_manifest.json").write_text(json.dumps(dict(files=[])), encoding="utf-8")

    monkeypatch.setattr(runner, "_snapshot_sources", snapshot)
    monkeypatch.setattr(runner, "EXTRA_SOURCES", ("configs/ppo_active_near_beta100.yaml",))
    output = tmp_path / "group"
    scenes = runner.prepare_group(config, output, historical, checked)
    assert runner.prepare_group(config, output, historical, checked) == scenes
    copied = output / "source_snapshot/configs/ppo_active_near_beta100.yaml"
    assert copied.read_bytes() == (ROOT / "configs/ppo_active_near_beta100.yaml").read_bytes()
    scenes["test_sha256"] = "mismatch"
    (output / "scenarios/scenario_manifest.json").write_text(json.dumps(scenes), encoding="utf-8")
    with pytest.raises(ValueError, match="scene sets"):
        runner.prepare_group(config, output, historical, checked)


def test_sequential_groups_acceptance_before_training_and_completed_reuse(tmp_path, monkeypatch):
    configs = configurations()
    before = deepcopy(configs)
    calls = []
    scenes = dict(test_angles=[0.0] * 500)

    def preflight(config, baseline):
        return dict(beta=config["environment_config"]["reward"]["best_so_far_shaping"]["beta"],
                    line_coefficient=0.01, historical_source_sha256={})

    def smoke(config, output):
        calls.append(("smoke", output.name))
        return dict(passed=True)

    def prepare(config, output, baseline, checked):
        output.mkdir(parents=True, exist_ok=True)
        return scenes

    def finish(config, output, seed):
        calls.append(("train", output.name, seed))
        post = output / ("seed_%d/postprocessing_001" % seed)
        return dict(status="completed", env_steps=300000, seed=seed), dict(status="completed", output_dir=str(post))

    def evaluate(checkpoint, output, angles, seed):
        calls.append(("evaluate", str(checkpoint)))
        assert angles == scenes["test_angles"]
        assert seed == 3001

    def analyze(output, group_outputs, baseline, posts):
        assert list(group_outputs) == ["beta100_k0p01", "beta200_k0p01"]
        assert all(len(values) == 3 for values in posts.values())
        report = output / "REPORT.md"
        report.write_text("Engineering fixture", encoding="utf-8")
        return report

    monkeypatch.setattr(runner, "preflight", preflight)
    monkeypatch.setattr(runner, "ensure_smoke", smoke)
    monkeypatch.setattr(runner, "prepare_group", prepare)
    monkeypatch.setattr(runner, "finish_seed", finish)
    monkeypatch.setattr(runner, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(runner, "verify_initialization", lambda *args: dict(passed=True))
    output = tmp_path / "suite"
    first = runner.run_experiment(configs, output, tmp_path / "baseline", analysis_fn=analyze)
    assert calls[:2] == [("smoke", "beta100_k0p01"), ("smoke", "beta200_k0p01")]
    assert [c[1:] for c in calls if c[0] == "train"] == [
        ("beta100_k0p01", 1001), ("beta100_k0p01", 1002), ("beta100_k0p01", 1003),
        ("beta200_k0p01", 1001), ("beta200_k0p01", 1002), ("beta200_k0p01", 1003)]
    assert len([c for c in calls if c[0] == "evaluate"]) == 12
    assert first["training_steps"] == 1800000
    old_calls = list(calls)
    second = runner.run_experiment(configs, output, tmp_path / "baseline", analysis_fn=analyze)
    assert second == first
    assert calls == old_calls
    assert configs == before


def test_smoke_only_does_not_enter_formal_group(tmp_path, monkeypatch):
    configs = configurations()
    monkeypatch.setattr(runner, "preflight", lambda *args: dict(historical_source_sha256={}))
    calls = []
    monkeypatch.setattr(runner, "ensure_smoke", lambda config, output: calls.append(output.name))
    monkeypatch.setattr(runner, "run_group", lambda *args: pytest.fail("Formal group started during acceptance"))
    result = runner.run_experiment(configs, tmp_path / "suite", tmp_path / "baseline", smoke_only=True)
    assert result["training_steps"] == 0
    assert calls == ["beta100_k0p01", "beta200_k0p01"]


def initial_checkpoint(value=1.0, trained=False):
    return dict(training_state={"env_steps": 0}, agent=dict(
        actor={"weight": torch.tensor([value])}, critic={"weight": torch.tensor([value + 1])},
        actor_optimizer={"state": {0: {"step": 1}} if trained else {}}, critic_optimizer={"state": {}}, update_count=0))


def test_initialization_verifies_weights_and_empty_adam(tmp_path):
    baseline = tmp_path / "baseline"
    groups = {"beta100": tmp_path / "beta100", "beta200": tmp_path / "beta200"}
    paths = [baseline / "seed_1001/checkpoints/initial.pt"]
    paths.extend(directory / "seed_1001/checkpoints/initial.pt" for directory in groups.values())
    for path in paths:
        path.parent.mkdir(parents=True)
        torch.save(initial_checkpoint(), path)
    assert runner.verify_initialization(groups, baseline, [1001])["passed"]
    torch.save(initial_checkpoint(value=2.0), paths[-1])
    with pytest.raises(RuntimeError, match="networks differ"):
        runner.verify_initialization(groups, baseline, [1001])
    torch.save(initial_checkpoint(trained=True), paths[-1])
    with pytest.raises(RuntimeError, match="Adam moments"):
        runner.verify_initialization(groups, baseline, [1001])


def test_partial_run_requires_explicit_recovery(tmp_path):
    config = configurations()[0]
    (tmp_path / "seed_1001").mkdir()
    with pytest.raises(RuntimeError, match="checkpoint recovery"):
        runner.finish_seed(config, tmp_path, 1001)
