"""Plotting fixtures: saved arrays and actual logged starts, without inference."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.plot_trajectory_evolution_with_initial_distribution import generate_plot, read_initial_distribution


def _csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "v09-r04_fixture"
    evolution = run / "trajectory_evolution"
    (evolution / "trajectories").mkdir(parents=True)
    stage = run / "stage_01_pm2"
    stage.mkdir()
    pursuers = [[1.5, 0, 1], [-0.75, 1.25, 1], [-0.75, -1.25, 1]]
    environment = dict(mission=dict(base_position=[0, 0, 0]), initialization=dict(pursuer_positions=pursuers))
    (run / "environment_config.yaml").write_text(yaml.safe_dump(environment), encoding="utf-8")
    (run / "metadata.json").write_text(json.dumps(dict(source_pretrained_env_steps=92160)), encoding="utf-8")
    # A planned future range must never appear as actual sampled starts.
    (run / "training_config.yaml").write_text(yaml.safe_dump(dict(curriculum=dict(stage_half_angles_degrees=[2, 5, 10]))), encoding="utf-8")
    starts = [dict(stage_index=1, half_angle_degrees=2.0, episode=i + 1, env_steps=3 * (i + 1),
                   angle_degrees=angle, initial_attacker_x=497.0, initial_attacker_y=angle * 8.0,
                   initial_attacker_z=50.0) for i, angle in enumerate([-1.0, 1.0])]
    _csv(stage / "training_metrics.csv", starts)
    initial = np.array(pursuers + [[497, 0, 50]], dtype=float)
    positions = np.stack((initial, initial + np.array([0.1, 0, 0])))
    metrics = dict(episode_length=1, captured=False, min_distance=450.0)
    records = []
    for step in [0, 512]:
        path = evolution / "trajectories" / ("step_{}.npz".format(step))
        np.savez(path, positions=positions, time=np.array([0, 0.1]), base_position=np.zeros(3),
                 attack_radius=np.array(10.0), metadata_json=np.array(json.dumps(dict(metrics=metrics))))
        records.append(dict(step=step, checkpoint=str(run / "invalid_model.pt"), trajectory=str(path), episode_length=1))
    _csv(evolution / "trajectory_evolution_metrics.csv", records)
    (evolution / "trajectory_evolution_manifest.json").write_text(json.dumps(dict(scene=dict(angle_degrees=0.0, seed=1000))), encoding="utf-8")
    (evolution / "trajectory_evolution_xy.png").write_bytes(b"old figure must be preserved")
    (run / "invalid_model.pt").write_bytes(b"not a torch checkpoint: plotting must not load it")
    return run


def test_only_logged_completed_starts_and_actual_stages_are_read(saved_run):
    starts, sources = read_initial_distribution(saved_run)
    assert len(starts) == 2 and len(sources) == 1
    assert {row["from_stage"] for row in starts} == {1}
    assert {row["half_angle_degrees"] for row in starts} == {2.0}
    assert [row["angle_degrees"] for row in starts] == [-1.0, 1.0]
    assert [row["initial_attacker_y"] for row in starts] == [-8.0, 8.0]


def test_plot_preserves_fixed_scene_sources_and_does_not_load_checkpoint(saved_run):
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in saved_run.rglob("*") if path.is_file()}
    result = generate_plot(saved_run)
    assert result["completed_episode_start_count"] == 2
    assert result["actual_stage_half_angles_degrees"] == [2.0]
    assert result["source_pretrained_env_steps"] == 92160
    assert result["fixed_test_scene"]["angle_degrees"] == 0.0
    assert result["selected_checkpoint_added_steps"] == [0, 512]
    assert result["all_original_sources_unchanged"]
    assert result["models_or_environments_loaded"] is False
    assert result["policy_inference_performed"] is False
    assert result["unfinished_episode_included"] is False
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in before.items())
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        generate_plot(saved_run)


def test_nonfinite_training_starts_are_rejected(saved_run):
    path = saved_run / "stage_01_pm2/training_metrics.csv"
    content = path.read_text(encoding="utf-8").replace("497.0", "nan", 1)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite initial"):
        read_initial_distribution(saved_run)
