"""A very short training run and the two standalone checkpoint consumers."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from algorithms.mappo import MAPPOAgent


ROOT = Path(__file__).resolve().parents[1]


def _run_module(module, *arguments):
    process_environment = os.environ.copy()
    process_environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    process_environment["MPLBACKEND"] = "Agg"
    process_environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-m", module, *map(str, arguments)],
        cwd=str(ROOT), env=process_environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", timeout=60, check=False,
    )
    assert result.returncode == 0, "{}\n{}".format(result.stdout, result.stderr)
    return result


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _check_trajectory(path):
    with np.load(path, allow_pickle=False) as trajectory:
        positions = trajectory["positions"]
        states = trajectory["states"]
        actions = trajectory["actions"]
        assert positions.ndim == 3 and positions.shape[1:] == (4, 3)
        assert states.shape == positions.shape[:2] + (6,)
        assert actions.shape == (positions.shape[0] - 1, 3, 3)
        assert trajectory["time"].shape == (positions.shape[0],)
        assert trajectory["rewards"].shape == (actions.shape[0],)
        assert trajectory["min_distance"].shape == (positions.shape[0],)
        assert trajectory["collision_pairs"].shape == (positions.shape[0],)
        assert actions.shape[0] <= 500
        assert np.all(np.abs(actions) <= 1.0)
        assert np.isfinite(positions).all()
        assert np.isfinite(states).all()
        assert np.isfinite(actions).all()
        np.testing.assert_allclose(positions, states[:, :, :3])
        with (ROOT / "configs" / "baseline_3a1c.yaml").open(encoding="utf-8") as stream:
            environment_config = yaml.safe_load(stream)
        initialization = environment_config["initialization"]
        np.testing.assert_allclose(
            positions[0], initialization["pursuer_positions"] + [initialization["attacker_position"]]
        )
        assert trajectory["base_position"].shape == (3,)
        assert float(trajectory["attack_radius"]) == environment_config["mission"]["attack_radius"]
        assert trajectory["distance_to_base"].shape == (positions.shape[0],)
        assert np.isfinite(trajectory["distance_to_base"]).all()


def test_short_training_evaluation_and_static_3d_visualization(tmp_path):
    with (ROOT / "configs" / "mappo_baseline.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["env_config"] = str(ROOT / "configs" / "baseline_3a1c.yaml")
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["training"]["rollout_steps"] = 8
    config["training"]["checkpoint_interval_updates"] = 1
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 4
    config["visualization"]["dpi"] = 60
    config_path = tmp_path / "training.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    training_dir = tmp_path / "training"
    _run_module(
        "scripts.train_mappo", "--config", config_path,
        "--total-steps", 16, "--output-dir", training_dir,
    )
    final_checkpoint = training_dir / "checkpoints" / "final.pt"
    best_checkpoint = training_dir / "checkpoints" / "best.pt"
    assert final_checkpoint.is_file()
    assert best_checkpoint.is_file()
    assert (training_dir / "checkpoints" / "update_000001.pt").is_file()
    assert (training_dir / "checkpoints" / "update_000002.pt").is_file()
    _, payload = MAPPOAgent.from_checkpoint(final_checkpoint, device="cpu")
    assert payload["training_state"]["env_steps"] == 16
    assert payload["training_state"]["update"] == 2
    assert payload["training_state"]["episodes"] == 0
    assert payload["training_state"]["partial_episode"]["episode_length"] == 16
    assert payload["training_state"]["completed_episodes_available"] is False
    _, best_payload = MAPPOAgent.from_checkpoint(best_checkpoint, device="cpu")
    assert best_payload["training_state"]["completed_episodes_available"] is False

    metadata = _read_json(training_dir / "metadata.json")
    assert metadata["seed"] == config["seed"]
    assert "code_version" in metadata
    assert metadata["output_dir"] == str(training_dir.resolve())
    assert metadata["environment_config"]["environment"]["max_episode_steps"] == 500
    assert (training_dir / "training_config.yaml").is_file()
    assert (training_dir / "environment_config.yaml").is_file()
    episode_fields, episode_rows = _read_csv(training_dir / "training_metrics.csv")
    assert {"episode_reward", "episode_length", "captured", "capture_rate", "min_distance", "collision_steps",
            "interception_success", "enemy_attack_success", "distance_to_base", "outcome"} <= set(episode_fields)
    assert episode_rows == []  # Sixteen steps cannot be reported as a full episode.
    update_fields, update_rows = _read_csv(training_dir / "updates.csv")
    assert {"actor_loss", "value_loss", "entropy", "env_steps"} <= set(update_fields)
    assert len(update_rows) == 2
    for row in update_rows:
        for field in ("actor_loss", "value_loss", "entropy"):
            assert np.isfinite(float(row[field]))

    evaluation_dir = tmp_path / "evaluation"
    _run_module(
        "scripts.evaluate_mappo", "--checkpoint", final_checkpoint,
        "--episodes", 1, "--output-dir", evaluation_dir,
    )
    fields, rows = _read_csv(evaluation_dir / "evaluation_metrics.csv")
    assert {
        "captured", "episode_reward", "episode_length", "capture_time",
        "min_distance", "collision_steps", "collision_step_rate",
        "average_pairwise_pursuer_distance",
        "interception_success", "enemy_attack_success", "truncated", "outcome",
        "attacker_base_terminal_distance", "interception_time",
    } <= set(fields)
    assert len(rows) == 1
    assert 1 <= int(rows[0]["episode_length"]) <= 500
    for field in ("episode_reward", "min_distance", "collision_step_rate"):
        assert np.isfinite(float(rows[0][field]))
    summary = _read_json(evaluation_dir / "evaluation_summary.json")
    assert summary["episodes"] == 1
    assert 0.0 <= summary["capture_rate"] <= 1.0
    assert 0.0 <= summary["collision_step_rate"] <= 1.0
    assert summary["capture_rate"] == summary["interception_success_rate"]
    assert sum(summary[name] for name in (
        "interception_success_rate", "enemy_attack_success_rate", "timeout_rate"
    )) == 1.0
    assert np.isfinite(summary["average_attacker_base_terminal_distance"])
    _check_trajectory(evaluation_dir / "first_episode.npz")

    visualization_dir = tmp_path / "visualization"
    _run_module(
        "scripts.visualize_mappo", "--checkpoint", final_checkpoint,
        "--output-dir", visualization_dir,
    )
    figure = visualization_dir / "trajectory_3d.png"
    assert figure.is_file()
    assert figure.stat().st_size > 1000
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    _check_trajectory(visualization_dir / "trajectory.npz")
    metrics = _read_json(visualization_dir / "visualization_metrics.json")
    assert "captured" in metrics
    assert 1 <= metrics["episode_length"] <= 500
