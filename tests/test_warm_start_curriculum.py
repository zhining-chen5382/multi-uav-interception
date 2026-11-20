"""Small engineering checks for checkpoint-initialized direction curricula.

Injected evaluation results exercise promotion only; they are not measured
interception performance. Source networks and Adam states come from real PPO.
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

from utils.experiment import set_seed


ROOT = Path(__file__).resolve().parents[1]
ROUTE = [0.0, 2.0, 5.0, 10.0, 30.0, 60.0, 90.0]
MODEL_KEYS = ("actor", "critic", "actor_optimizer", "critic_optimizer", "rng_state")


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _summary(episodes, capture=0.0, capture_time=1.0):
    return dict(episodes=episodes, capture_rate=capture,
        enemy_attack_success_rate=1.0 - capture,
        average_capture_time=capture_time if capture else None,
        average_min_distance=4.0 if capture else 40.0,
        collision_episode_rate=0.0)


def _failure(agent, environment, half_angle, episodes, seed, directory):
    return _summary(episodes)


@pytest.fixture
def warm_config(tmp_path, monkeypatch):
    from scripts import train_direction_curriculum, train_mappo

    monkeypatch.setattr(train_direction_curriculum, "run_post_training_analysis", lambda *args: {"status": "disabled"})
    monkeypatch.setattr(train_mappo, "run_post_training_analysis", lambda *args: {"status": "disabled"})
    config = _yaml(ROOT / "configs/mappo_direction_curriculum_zero_start.yaml")
    environment = _yaml(ROOT / "configs/baseline_capture500.yaml")
    environment["environment"]["max_episode_steps"] = 3
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(environment, sort_keys=False), encoding="utf-8")
    config["env_config"] = str(path)
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 2
    config["training"]["rollout_steps"] = 3
    config["training"]["checkpoint_interval_updates"] = 1
    config["curriculum"]["max_stage_steps"] = 12
    config["curriculum"]["evaluation_interval_steps"] = 4
    config["curriculum"]["evaluation_episodes"] = 4
    config["curriculum"]["evaluation_timing"] = "after_full_rollout"
    config["training"]["total_env_steps"] = 12 * len(ROUTE)

    source_config = deepcopy(config)
    source_config.pop("curriculum")
    source_config["training"]["total_env_steps"] = 5
    source_config["output_dir"] = str(tmp_path / "source")
    source_path = tmp_path / "source_training.yaml"
    source_path.write_text(yaml.safe_dump(source_config, sort_keys=False), encoding="utf-8")
    train_mappo.train(source_path)
    checkpoint = tmp_path / "source/checkpoints/final.pt"
    original = _payload(checkpoint)
    assert original["actor_optimizer"]["state"] and original["critic_optimizer"]["state"]
    assert original["training_state"]["env_steps"] == 5
    assert original["training_state"]["partial_episode"]["episode_length"] == 2
    config["warm_start"] = dict(checkpoint=str(checkpoint), expected_sha256=_sha(checkpoint))
    return config


def _train(tmp_path, config, evaluator=_failure, name="warm_run"):
    from scripts.train_direction_curriculum import train_curriculum

    config = deepcopy(config)
    output = tmp_path / name
    config["output_dir"] = str(output)
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output, train_curriculum(path, evaluation_fn=evaluator)


def _stage(output, number=1, half_angle=0):
    return Path(output) / "stage_{:02d}_pm{:g}".format(number, half_angle)


def test_warm_source_loading_is_read_only_and_does_not_consume_rng(warm_config):
    from utils.curriculum_warm_start import load_warm_start_source

    checkpoint = Path(warm_config["warm_start"]["checkpoint"])
    digest = _sha(checkpoint)
    set_seed(19, 1)
    before = dict(python=random.getstate(), numpy=deepcopy(np.random.get_state()), torch=torch.get_rng_state().clone())
    context = load_warm_start_source(warm_config, _yaml(warm_config["env_config"]))
    assert Path(context["checkpoint"]) == checkpoint
    assert context["source_env_steps"] == 5
    assert context["discarded_partial_episode_steps"] == 2
    assert context["sha256"] == digest == _sha(checkpoint)
    _equal(random.getstate(), before["python"])
    _equal(np.random.get_state(), before["numpy"])
    _equal(torch.get_rng_state(), before["torch"])


def test_warm_initial_inherits_models_adam_rng_and_starts_a_new_episode(tmp_path, warm_config):
    source_path = Path(warm_config["warm_start"]["checkpoint"])
    source = _payload(source_path)
    source_hash = _sha(source_path)
    seen = []

    def evaluate(agent, environment, half_angle, episodes, seed, directory):
        if Path(directory).parent.name == "at_000000":
            seen.append({name: deepcopy(getattr(agent, name).state_dict())
                         for name in MODEL_KEYS if name != "rng_state"})
        return _summary(episodes)

    output, result = _train(tmp_path, warm_config, evaluate)
    initial = _payload(_stage(output) / "checkpoints/initial.pt")
    for name in MODEL_KEYS:
        _equal(initial[name], source[name])
    assert len(seen) == 1
    for name, actual in seen[0].items():
        _equal(actual, source[name])
    state = initial["training_state"]
    assert state["env_steps"] == state["update"] == state["episodes"] == 0
    assert state["source_pretrained_env_steps"] == state["lineage_env_steps"] == 5
    assert state["environment_step_count"] == state["partial_episode"]["episode_length"] == 0
    assert state["partial_episode"]["episode_reward"] == 0
    np.testing.assert_array_equal(state["environment_accelerations"], np.zeros((4, 3)))
    environment = _yaml(warm_config["env_config"])
    expected = np.vstack((environment["initialization"]["pursuer_positions"],
                          environment["initialization"]["attacker_position"]))
    np.testing.assert_array_equal(np.asarray(state["environment_state"]).reshape(4, 6)[:, :3], expected)
    assert result["env_steps"] == 12 and len(result["stages"]) == 1
    assert _sha(source_path) == source_hash


def test_initial_success_is_a_best_candidate_but_does_not_satisfy_promotion(tmp_path, warm_config):
    calls = []

    def evaluate(agent, environment, half_angle, episodes, seed, directory):
        initial = Path(directory).parent.name == "at_000000"
        stage = next(part for part in Path(directory).parts if part.startswith("stage_") and "_pm" in part)
        calls.append((stage, initial))
        # Baseline and first post-training checkpoint succeed, then failures.
        capture = 1.0 if initial or len(calls) == 2 else 0.0
        return _summary(episodes, capture=capture, capture_time=1.0 if initial else 2.0)

    output, result = _train(tmp_path, warm_config, evaluate)
    assert len(result["stages"]) == 1 and result["status"] == "stopped_stage_budget"
    assert not _stage(output, 2, 2).exists()
    assert result["stages"][0]["evaluations"] == 3
    history = _rows(output / "evaluation_history.csv")
    initial = next(row for row in history if row["role"] == "initial")
    current = [row for row in history if row["role"] == "current"]
    assert initial["stage_env_steps"] == "0" and int(initial["consecutive_passes"]) == 0
    assert [int(row["consecutive_passes"]) for row in current] == [1, 0, 0]
    best = _payload(_stage(output) / "checkpoints/best_eval.pt")
    assert best["training_state"]["env_steps"] == 0
    assert best["training_state"]["evaluation_summary"]["capture_rate"] == 1.0


def test_after_full_rollout_evaluates_crossings_and_retains_all_samples(tmp_path, warm_config):
    output, result = _train(tmp_path, warm_config)
    assert result["env_steps"] == 12
    updates = _rows(_stage(output) / "updates.csv")
    assert [int(row["rollout_steps"]) for row in updates] == [3, 3, 3, 3]
    assert [int(row["env_steps"]) for row in updates] == [3, 6, 9, 12]
    current = [row for row in _rows(output / "evaluation_history.csv") if row["role"] == "current"]
    assert [int(row["stage_env_steps"]) for row in current] == [6, 9, 12]


def test_stage_hard_budget_tail_is_evaluated_once(tmp_path, warm_config):
    config = deepcopy(warm_config)
    config["curriculum"]["max_stage_steps"] = 16
    config["training"]["total_env_steps"] = 16 * len(ROUTE)
    output, result = _train(tmp_path, config)
    updates = _rows(_stage(output) / "updates.csv")
    assert result["env_steps"] == 16
    assert [int(row["rollout_steps"]) for row in updates] == [3, 3, 3, 3, 3, 1]
    current = [row for row in _rows(output / "evaluation_history.csv") if row["role"] == "current"]
    assert [int(row["stage_env_steps"]) for row in current] == [6, 9, 12, 16]
    assert int(current[-1]["env_steps"]) == 16


def test_promotion_inherits_historical_initial_best_instead_of_later_weights(tmp_path, warm_config):
    def evaluate(agent, environment, half_angle, episodes, seed, directory):
        # Zero passes twice after training, but its baseline remains fastest.
        if half_angle == 0:
            return _summary(episodes, capture=1.0,
                            capture_time=1.0 if Path(directory).parent.name == "at_000000" else 2.0)
        return _summary(episodes)

    output, result = _train(tmp_path, warm_config, evaluate)
    assert len(result["stages"]) == 2
    assert [stage["stage_env_steps"] for stage in result["stages"]] == [9, 12]
    assert result["env_steps"] == 21
    best_path = _stage(output) / "checkpoints/best_eval.pt"
    best = _payload(best_path)
    last = _payload(_stage(output) / "checkpoints/last.pt")
    inherited = _payload(_stage(output, 2, 2) / "checkpoints/initial.pt")
    assert best["training_state"]["env_steps"] == 0 and last["training_state"]["env_steps"] == 9
    assert any(not torch.equal(last["actor"][key], best["actor"][key]) for key in best["actor"])
    for key in MODEL_KEYS:
        _equal(inherited[key], best[key])
    assert inherited["training_state"]["env_steps"] == 9
    assert result["stages"][1]["parent_checkpoint_sha256"] == _sha(best_path)
    # Excellent zero-angle forgetting monitoring never promotes failing +/-2.
    history = _rows(output / "evaluation_history.csv")
    previous = [row for row in history if row["role"] == "previous"]
    assert len(previous) == 3 and all(float(row["capture_rate"]) == 1.0 for row in previous)
    assert all(row["consecutive_passes"] == "" for row in previous)
    assert not _stage(output, 3, 5).exists()


def test_scheduler_keeps_real_512_step_batch_at_the_10000_threshold():
    from utils.curriculum_warm_start import evaluation_due, rollout_collection_steps

    curriculum = dict(evaluation_timing="after_full_rollout", evaluation_interval_steps=10000)
    assert rollout_collection_steps(curriculum, 9728, 100000, 700000 - 9728, 512) == 512
    assert evaluation_due(curriculum, 9728, 10240, 100000)
    assert not evaluation_due(curriculum, 10240, 10752, 100000)
    assert rollout_collection_steps(curriculum, 99840, 100000, 700000 - 99840, 512) == 160
    assert evaluation_due(curriculum, 99840, 100000, 100000)


def test_scheduler_keeps_legacy_exact_step_mode_and_total_budget_final_evaluation():
    from utils.curriculum_warm_start import evaluation_due, rollout_collection_steps

    curriculum = dict(evaluation_interval_steps=10000)
    assert rollout_collection_steps(curriculum, 9728, 100000, 700000, 512) == 272
    assert evaluation_due(curriculum, 9728, 10000, 100000)
    curriculum["evaluation_timing"] = "after_full_rollout"
    assert rollout_collection_steps(curriculum, 10240, 100000, 160, 512) == 160
    assert evaluation_due(curriculum, 10240, 10400, 100000, total_budget_exhausted=True)


@pytest.mark.parametrize("change", ["hash", "network", "reward", "missing_optimizer", "nonfinite", "missing_checkpoint"])
def test_invalid_warm_sources_are_rejected_before_output_creation(tmp_path, warm_config, change):
    config = deepcopy(warm_config)
    if change == "hash":
        config["warm_start"]["expected_sha256"] = "0" * 64
    elif change == "network":
        config["network"]["actor_hidden_sizes"] = [8, 8]
    elif change == "reward":
        environment = _yaml(config["env_config"])
        environment["reward"]["capture_reward"] += 1.0
        path = tmp_path / "changed_environment.yaml"
        path.write_text(yaml.safe_dump(environment), encoding="utf-8")
        config["env_config"] = str(path)
    elif change == "missing_checkpoint":
        config["warm_start"]["checkpoint"] = str(tmp_path / "absent.pt")
    else:
        payload = _payload(config["warm_start"]["checkpoint"])
        if change == "missing_optimizer":
            payload["actor_optimizer"]["state"] = {}
        else:
            key = next(name for name, value in payload["actor"].items() if value.dtype.is_floating_point)
            payload["actor"][key].reshape(-1)[0] = float("nan")
        path = tmp_path / "invalid.pt"
        torch.save(payload, path)
        config["warm_start"] = dict(checkpoint=str(path), expected_sha256=_sha(path))
    with pytest.raises((ValueError, FileNotFoundError)):
        _train(tmp_path, config, name="invalid_output")
    assert not (tmp_path / "invalid_output").exists()


@pytest.mark.parametrize("timing", ["unknown", True, None])
def test_invalid_evaluation_timing_rejected_before_output_creation(tmp_path, warm_config, timing):
    config = deepcopy(warm_config)
    config["curriculum"]["evaluation_timing"] = timing
    with pytest.raises(ValueError):
        _train(tmp_path, config, name="invalid_output")
    assert not (tmp_path / "invalid_output").exists()


def test_evaluation_interval_shorter_than_full_rollout_is_rejected(tmp_path, warm_config):
    config = deepcopy(warm_config)
    config["training"]["rollout_steps"] = 5
    with pytest.raises(ValueError):
        _train(tmp_path, config, name="invalid_output")
    assert not (tmp_path / "invalid_output").exists()


@pytest.mark.parametrize("conflict", ["continuation", "initial_checkpoint", "resume_checkpoint"])
def test_warm_start_rejects_other_initialization_or_resume_modes(tmp_path, warm_config, conflict):
    config = deepcopy(warm_config)
    config[conflict] = {"source_output": str(tmp_path / "source")} if conflict == "continuation" else config["warm_start"]["checkpoint"]
    with pytest.raises(ValueError):
        _train(tmp_path, config, name="invalid_output")
    assert not (tmp_path / "invalid_output").exists()


def _positive_curriculum(config):
    result = deepcopy(config)
    result["curriculum"]["stage_half_angles_degrees"] = ROUTE[1:]
    result["training"]["total_env_steps"] = result["curriculum"]["max_stage_steps"] * 6
    result["warm_start"]["previous_stage_half_angle_degrees"] = 0.0
    return result


def test_formal_warm_configuration_enters_two_degrees_and_keeps_v08_learning_settings():
    from scripts.train_direction_curriculum import validate_config

    config = _yaml(ROOT / "configs/mappo_direction_curriculum_warm_start.yaml")
    reference = _yaml(ROOT / "configs/mappo_capture500.yaml")
    assert config["curriculum"]["stage_half_angles_degrees"] == ROUTE[1:]
    assert config["training"]["total_env_steps"] == 600000
    assert config["curriculum"]["max_stage_steps"] == 100000
    assert config["curriculum"]["evaluation_timing"] == "after_full_rollout"
    assert config["curriculum"]["evaluation_interval_steps"] == 10000
    assert config["curriculum"]["evaluation_episodes"] == 100
    assert config["curriculum"]["required_consecutive_evaluations"] == 2
    assert config["curriculum"]["promotion_capture_rate"] == 0.8
    assert config["warm_start"]["previous_stage_half_angle_degrees"] == 0.0
    assert config["warm_start"]["checkpoint"].endswith("v09-r03_20261109_seed000/checkpoints/update_000180.pt")
    assert len(config["warm_start"]["expected_sha256"]) == 64
    assert not config.get("continuation") and not config.get("initial_checkpoint")
    for key in ("network", "ppo", "normalization", "env_config", "seed", "device", "torch_num_threads"):
        assert config[key] == reference[key]
    for key in ("rollout_steps", "log_interval_updates", "checkpoint_interval_updates", "capture_rate_window"):
        assert config["training"][key] == reference["training"][key]
    validate_config(config)


def test_first_positive_stage_monitors_source_zero_without_counting_it_for_promotion(tmp_path, warm_config):
    config = _positive_curriculum(warm_config)
    calls = []

    def evaluate(agent, environment, half_angle, episodes, seed, directory):
        calls.append((half_angle, episodes, Path(directory).name))
        return _summary(episodes, capture=1.0 if half_angle == 0 else 0.79)

    output, result = _train(tmp_path, config, evaluate)
    assert len(result["stages"]) == 1 and result["status"] == "stopped_stage_budget"
    assert result["stages"][0]["half_angle_degrees"] == 2
    assert result["stages"][0]["evaluations"] == 3
    assert calls == [(2.0, 4, "current"), (0.0, 1, "previous")] * 4
    history = _rows(output / "evaluation_history.csv")
    assert [row["role"] for row in history[:2]] == ["initial", "initial_previous"]
    current = [row for row in history if row["role"] == "current"]
    assert [int(row["stage_env_steps"]) for row in current] == [6, 9, 12]
    assert all(int(row["consecutive_passes"]) == 0 for row in current)
    assert not _stage(output, 2, 5).exists()
    # Real training still samples the new incoming-angle distribution.
    episode_rows = _rows(_stage(output, 1, 2) / "training_metrics.csv")
    angles = [float(row["angle_degrees"]) for row in episode_rows]
    assert len(set(angles)) > 1 and all(-2 <= angle <= 2 for angle in angles)
    assert any(angle != 0 for angle in angles)


def test_first_positive_initial_and_source_monitor_do_not_replace_two_current_passes(tmp_path, warm_config):
    config = _positive_curriculum(warm_config)
    current_count = 0

    def evaluate(agent, environment, half_angle, episodes, seed, directory):
        nonlocal current_count
        if half_angle == 0 or Path(directory).parent.name == "at_000000":
            return _summary(episodes, capture=1.0)
        current_count += 1
        return _summary(episodes, capture=0.8 if current_count == 1 else 0.0)

    output, result = _train(tmp_path, config, evaluate)
    assert len(result["stages"]) == 1
    current = [row for row in _rows(output / "evaluation_history.csv") if row["role"] == "current"]
    assert [int(row["consecutive_passes"]) for row in current] == [1, 0, 0]
    assert not _stage(output, 2, 5).exists()


def test_positive_stage_promotes_from_historical_best_with_complete_adam_and_rng(tmp_path, warm_config):
    config = _positive_curriculum(warm_config)

    def evaluate(agent, environment, half_angle, episodes, seed, directory):
        if half_angle == 0:
            return _summary(episodes, capture=1.0)
        if half_angle == 2:
            return _summary(episodes, capture=1.0 if Path(directory).parent.name == "at_000000" else 0.8)
        return _summary(episodes)

    output, result = _train(tmp_path, config, evaluate)
    assert len(result["stages"]) == 2
    assert [stage["half_angle_degrees"] for stage in result["stages"]] == [2.0, 5.0]
    assert [stage["stage_env_steps"] for stage in result["stages"]] == [9, 12]
    best_path = _stage(output, 1, 2) / "checkpoints/best_eval.pt"
    best = _payload(best_path)
    inherited = _payload(_stage(output, 2, 5) / "checkpoints/initial.pt")
    assert best["training_state"]["env_steps"] == 0
    assert inherited["training_state"]["env_steps"] == 9
    for name in MODEL_KEYS:
        _equal(inherited[name], best[name])
    assert result["stages"][1]["parent_checkpoint_sha256"] == _sha(best_path)
    assert not _stage(output, 3, 10).exists()
