"""Frozen-policy trajectory reports and post-training callback integration.

All episodes use tiny networks and at most three simulator steps. The short
training tests only verify callback placement, not research performance.
"""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
import yaml

from algorithms.mappo import MAPPOAgent, RolloutBuffer
from envs import PursuitEnv
from utils.experiment import set_seed


ROOT = Path(__file__).resolve().parents[1]


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and torch.equal(actual, expected)
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
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob("*") if p.is_file()}


def _global_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(), threads=torch.get_num_threads(),
                deterministic=torch.backends.cudnn.deterministic,
                benchmark=torch.backends.cudnn.benchmark,
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


@pytest.fixture(autouse=True)
def preserve_process_random_state():
    before = _global_state()
    yield
    random.setstate(before["python"])
    np.random.set_state(before["numpy"])
    torch.set_rng_state(before["torch"])
    torch.set_num_threads(before["threads"])
    torch.backends.cudnn.deterministic = before["deterministic"]
    torch.backends.cudnn.benchmark = before["benchmark"]
    if before["cuda"] is not None:
        torch.cuda.set_rng_state_all(before["cuda"])


@pytest.fixture
def small_config(tmp_path):
    environment = _yaml(ROOT / "configs/baseline_capture500.yaml")
    environment["environment"]["max_episode_steps"] = 3
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(environment), encoding="utf-8")
    config = _yaml(ROOT / "configs/mappo_capture500.yaml")
    config["env_config"] = str(path)
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 2
    config["training"]["rollout_steps"] = 4
    config["training"]["log_interval_updates"] = 1
    config["training"]["checkpoint_interval_updates"] = 1
    return config, environment, path


def _one_update(agent, environment_path):
    env = PursuitEnv(environment_path)
    observations, info = env.reset(seed=3)
    buffer = RolloutBuffer(4, 3, 21, 24, 3)
    for _ in range(4):
        actions, log_probs, latent, value = agent.act(observations, info["global_state"])
        next_observations, rewards, terminal, truncated, next_info = env.step(actions)
        next_value = 0.0 if terminal else agent.value(next_info["global_state"])
        buffer.add(observations, info["global_state"], latent, log_probs, float(rewards[0]),
                   value, next_value, terminal, truncated)
        observations, info = next_observations, next_info
        if terminal or truncated:
            observations, info = env.reset()
    buffer.compute_returns(agent.config["ppo"]["gamma"], agent.config["ppo"]["gae_lambda"])
    agent.update(buffer)


def _saved_run(tmp_path, small_config, *, stage=False, steps=(5, 12, 20, 30)):
    config, environment, environment_path = small_config
    run = tmp_path / "run"
    checkpoint_dir = run / ("stage_01_pm10/checkpoints" if stage else "checkpoints")
    checkpoint_dir.mkdir(parents=True)
    set_seed(25, 1)
    agent = MAPPOAgent(config)
    if steps[0] != 0:
        _one_update(agent, environment_path)
        assert agent.actor_optimizer.state_dict()["state"]
        assert agent.critic_optimizer.state_dict()["state"]
    for index, step in enumerate(steps):
        if index:
            _one_update(agent, environment_path)
        # Deliberately use filenames unrelated to their actual step counters.
        name = "update_999999.pt" if index == 0 else "policy_{}.pt".format(index)
        if index == len(steps) - 1:
            name = "last.pt" if stage else "final.pt"
        agent.save(checkpoint_dir / name, training_state={"env_steps": step, "update": index + 1},
                   metadata={"environment_config": deepcopy(environment)})
        if step == 20:
            shutil.copyfile(checkpoint_dir / name, checkpoint_dir / ("best_eval.pt" if stage else "best.pt"))
    (run / "environment_config.yaml").write_text(yaml.safe_dump(environment), encoding="utf-8")
    (run / "training_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run


@pytest.fixture
def evolution_path(tmp_path):
    config = _yaml(ROOT / "configs/trajectory_evolution.yaml")
    config.update(target_steps=[0, 30], extra_checkpoint_steps=[], angle_degrees=0.0,
                  seed=1000, dpi=60, figure_size=[8, 6])
    path = tmp_path / "evolution.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("stage", [False, True])
def test_actual_checkpoint_steps_nearest_sampling_aliases_and_explicit_extra(
    tmp_path, small_config, stage
):
    from utils.trajectory_evolution import discover_checkpoints, select_checkpoints

    run = _saved_run(tmp_path, small_config, stage=stage)
    # The discovery must not recurse into snapshots, videos or other runs.
    for directory in ("source_snapshot/checkpoints", "videos/checkpoints", "continuation_other/checkpoints"):
        hidden = run / directory
        hidden.mkdir(parents=True)
        (hidden / "not_a_checkpoint.pt").write_bytes(b"not a checkpoint")
    checkpoints = discover_checkpoints(run)
    assert [row["step"] for row in checkpoints] == [5, 12, 20, 30]
    best = next(row for row in checkpoints if row["step"] == 20)
    assert len(best["aliases"]) >= 2
    selected, mapping = select_checkpoints(checkpoints, [0, 30], extra_steps=[12])
    assert [row["step"] for row in selected] == [5, 12, 20, 30]
    assert len({row["step"] for row in selected}) == len(selected)
    assert next(row for row in mapping if row["requested_step"] == 0)["actual_step"] == 5
    assert 0 not in [row["step"] for row in selected]


def test_same_step_with_conflicting_weights_is_rejected(tmp_path, small_config):
    from utils.trajectory_evolution import discover_checkpoints

    run = _saved_run(tmp_path, small_config)
    original = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    name = next(iter(original["actor"]))
    original["actor"][name] = original["actor"][name] + 0.1
    torch.save(original, run / "checkpoints/conflicting.pt")
    with pytest.raises(ValueError):
        discover_checkpoints(run)


def test_frozen_full_episodes_csv_and_npz_preserve_sources_and_global_state(
    tmp_path, small_config, evolution_path, monkeypatch
):
    from utils.trajectory_evolution import generate_trajectory_evolution

    run = _saved_run(tmp_path, small_config)
    hashes_before = _hash_tree(run)

    def unexpected_update(*args, **kwargs):
        raise AssertionError("Trajectory reporting must never update a policy")

    monkeypatch.setattr(MAPPOAgent, "update", unexpected_update)
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    before = _global_state()
    manifest = generate_trajectory_evolution(run, config_path=evolution_path, extra_steps=[12])
    _equal(_global_state(), before)
    output = run / "trajectory_evolution"
    assert manifest["selected_steps"] == [5, 12, 20, 30]
    assert manifest["initial_checkpoint_available"] is False
    assert manifest["zero_step_note"]
    assert manifest["deterministic"] is True
    assert manifest["source_files_unchanged"] is True
    metrics = _rows(output / "trajectory_evolution_metrics.csv")
    assert [int(row["step"]) for row in metrics] == [5, 12, 20, 30]
    initial_states = []
    base = np.asarray(small_config[1]["mission"]["base_position"])
    for row in metrics:
        step = int(row["step"])
        with np.load(output / "trajectories" / ("step_{:06d}.npz".format(step)), allow_pickle=False) as archive:
            states, times = archive["states"], archive["time"]
            initial_states.append(states[0])
            assert states.shape == (4, 4, 6)
            assert archive["actions"].shape == (3, 3, 3)
            assert np.isfinite(states).all() and np.isfinite(archive["actions"]).all()
            assert len(times) == int(row["episode_length"]) + 1
            assert int(row["episode_length"]) == 3 and row["result"] == "Timeout"
            assert float(row["event_time"]) == pytest.approx(0.3)
            distances = np.linalg.norm(states[:, :3, :3] - states[:, 3:4, :3], axis=-1)
            assert float(row["min_pursuer_attacker_distance"]) == pytest.approx(distances.min())
            assert float(row["final_attacker_base_distance"]) == pytest.approx(np.linalg.norm(states[-1, 3, :3] - base))
            saved = json.loads(archive["metadata_json"].item())
            assert saved["seed"] == 1000
    assert all(np.array_equal(initial_states[0], state) for state in initial_states[1:])
    for name, digest in hashes_before.items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
    plots = list(output.glob("*.png"))
    assert any("xy" in path.name.lower() for path in plots)
    assert all(path.stat().st_size > 1000 for path in plots)
    assert _json(output / "trajectory_evolution_manifest.json")["selected_steps"] == manifest["selected_steps"]


@pytest.mark.parametrize("attacker_position,result,captured", [
    ([1.5, 0.0, 4.0], "Capture", True),
    ([0.0, 0.0, 9.0], "Fail", False),
])
def test_csv_records_actual_terminal_event_and_three_dimensional_distances(
    tmp_path, small_config, evolution_path, attacker_position, result, captured
):
    from utils.trajectory_evolution import generate_trajectory_evolution

    config, environment, path = small_config
    environment["initialization"]["attacker_position"] = attacker_position
    path.write_text(yaml.safe_dump(environment), encoding="utf-8")
    run = _saved_run(tmp_path, (config, environment, path), steps=(0,))
    generate_trajectory_evolution(run, config_path=evolution_path)
    output = run / "trajectory_evolution"
    row = _rows(output / "trajectory_evolution_metrics.csv")[0]
    assert int(row["step"]) == 0 and row["result"] == result
    assert row["captured"] in (str(captured), str(int(captured)))
    assert int(row["episode_length"]) == 1 and float(row["event_time"]) == pytest.approx(0.1)
    with np.load(output / "trajectories/step_000000.npz", allow_pickle=False) as archive:
        states = archive["states"]
        distances = np.linalg.norm(states[:, :3, :3] - states[:, 3:4, :3], axis=-1)
        assert float(row["min_pursuer_attacker_distance"]) == pytest.approx(distances.min())
        assert distances.min() > np.linalg.norm(states[:, :3, :2] - states[:, 3:4, :2], axis=-1).min() + 1


def test_repeated_report_matches_trajectories_and_refuses_output_overwrite(
    tmp_path, small_config, evolution_path
):
    from utils.trajectory_evolution import generate_trajectory_evolution

    run = _saved_run(tmp_path, small_config, steps=(0, 30))
    first, second = tmp_path / "first", tmp_path / "second"
    generate_trajectory_evolution(run, config_path=evolution_path, output_dir=first)
    generate_trajectory_evolution(run, config_path=evolution_path, output_dir=second)
    for step in (0, 30):
        name = "trajectories/step_{:06d}.npz".format(step)
        with np.load(first / name, allow_pickle=False) as left, np.load(second / name, allow_pickle=False) as right:
            for field in ("states", "actions", "time", "rewards"):
                np.testing.assert_array_equal(left[field], right[field])
    before = _hash_tree(first)
    with pytest.raises((ValueError, FileExistsError)):
        generate_trajectory_evolution(run, config_path=evolution_path, output_dir=first)
    assert _hash_tree(first) == before


@pytest.mark.parametrize("field,value", [
    ("enabled", "yes"), ("seed", -1), ("angle_degrees", float("nan")),
    ("target_steps", [-1]), ("target_steps", [1.25]), ("dpi", 0),
])
def test_invalid_reporting_configuration_is_rejected(tmp_path, evolution_path, field, value):
    from utils.trajectory_evolution import load_evolution_config

    options = _yaml(evolution_path)
    options[field] = value
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(options), encoding="utf-8")
    with pytest.raises(ValueError):
        load_evolution_config(invalid)


def test_mixed_checkpoint_environment_is_rejected_before_output_creation(
    tmp_path, small_config, evolution_path
):
    from utils.trajectory_evolution import generate_trajectory_evolution

    run = _saved_run(tmp_path, small_config)
    checkpoint = run / "checkpoints/final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["metadata"]["environment_config"]["capture"]["radius"] = 6.0
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        generate_trajectory_evolution(run, config_path=evolution_path)
    assert not (run / "trajectory_evolution").exists()


def test_ordinary_training_calls_report_only_after_initial_and_final_checkpoints_exist(
    tmp_path, small_config, monkeypatch
):
    from scripts import train_mappo

    config, _, _ = small_config
    config["training"]["total_env_steps"] = 2
    config["output_dir"] = str(tmp_path / "ordinary")
    path = tmp_path / "ordinary.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []

    def completed(output_dir, training_config):
        output = Path(output_dir)
        initial = torch.load(output / "checkpoints/initial.pt", map_location="cpu", weights_only=False)
        final = torch.load(output / "checkpoints/final.pt", map_location="cpu", weights_only=False)
        assert initial["training_state"]["env_steps"] == 0
        assert final["training_state"]["env_steps"] == 2
        calls.append((output, training_config))

    monkeypatch.setattr(train_mappo, "run_post_training_analysis", completed)
    result = train_mappo.train(path)
    assert result["env_steps"] == 2 and len(calls) == 1
    assert calls[0][0] == Path(config["output_dir"]).resolve()
    assert calls[0][1]["network"] == config["network"]


def test_curriculum_calls_report_after_terminal_summary_and_last_checkpoint_exist(
    tmp_path, small_config, monkeypatch
):
    from scripts import train_direction_curriculum

    config, _, _ = small_config
    config["curriculum"] = _yaml(ROOT / "configs/mappo_direction_curriculum.yaml")["curriculum"]
    config["curriculum"].update(max_stage_steps=4, evaluation_interval_steps=2, evaluation_episodes=4)
    config["training"]["total_env_steps"] = 16
    config["output_dir"] = str(tmp_path / "curriculum")
    path = tmp_path / "curriculum.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []

    def completed(output_dir, training_config):
        output = Path(output_dir)
        terminal = _json(output / "summary.json")
        last = torch.load(output / "stage_01_pm10/checkpoints/last.pt", map_location="cpu", weights_only=False)
        assert terminal["env_steps"] == last["training_state"]["env_steps"] == 4
        calls.append((output, training_config))

    def evaluate(agent, env_config, half_angle, episodes, seed, output_dir):
        return dict(episodes=episodes, capture_rate=0.0, enemy_attack_success_rate=1.0,
                    average_capture_time=None, average_min_distance=100.0, collision_episode_rate=0.0)

    monkeypatch.setattr(train_direction_curriculum, "run_post_training_analysis", completed)
    result = train_direction_curriculum.train_curriculum(path, evaluation_fn=evaluate)
    assert result["status"] == "stopped_stage_budget" and len(calls) == 1
    assert calls[0][0] == Path(config["output_dir"]).resolve()


def test_post_training_can_be_explicitly_disabled_without_artifacts(tmp_path):
    from utils.post_training import run_post_training_analysis

    output = tmp_path / "disabled"
    output.mkdir()
    options = tmp_path / "disabled.yaml"
    options.write_text(yaml.safe_dump({"enabled": False}), encoding="utf-8")
    before = _hash_tree(output)
    result = run_post_training_analysis(output, {"post_training": {"trajectory_evolution_config": str(options)}})
    assert result == {"status": "disabled"}
    assert _hash_tree(output) == before


def test_post_training_failure_warns_and_preserves_saved_checkpoints(tmp_path, monkeypatch):
    from utils import trajectory_evolution
    from utils.post_training import run_post_training_analysis

    output = tmp_path / "failed"
    output.mkdir()
    checkpoint = output / "final.pt"
    checkpoint.write_bytes(b"saved model must survive a reporting error")
    before = checkpoint.read_bytes()
    options = tmp_path / "enabled.yaml"
    options.write_text(yaml.safe_dump({"enabled": True, "directory_name": "trajectory_evolution"}), encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate plotting failure")

    monkeypatch.setattr(trajectory_evolution, "generate_trajectory_evolution", fail)
    with pytest.warns(RuntimeWarning, match="trajectory evolution failed"):
        result = run_post_training_analysis(output, {"post_training": {"trajectory_evolution_config": str(options)}})
    assert result["status"] == "failed" and result["training_checkpoints_preserved"] is True
    assert _json(output / "trajectory_evolution_error.json")["error"] == "deliberate plotting failure"
    assert checkpoint.read_bytes() == before
