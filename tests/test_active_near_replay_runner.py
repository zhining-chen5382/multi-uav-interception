"""Regression coverage for paired offline replay accounting."""

from copy import deepcopy
import json

import numpy as np
import pytest
import yaml

from scripts.offline_active_near_replay import replay_episode, summarize, run
from utils.active_near_replay import active_near_shaping


def episode(moving=False, cohort="test"):
    states = np.zeros((3, 2, 6))
    states[:, 1, 0] = [20.0, 18.0, 16.0]
    if moving:
        states[:, 0, 0] = [0.0, 1.0, 2.0]
    near = active_near_shaping(states, 20.0, 100.0)
    return dict(episode_id="sample", dataset="reference_B", cohort=cohort,
                training_seed=1001, model="best", checkpoint_step=10000,
                scenario_id=0, environment_seed=3001, angle_degrees=0.0,
                category="near_miss", duplicate_of="", states=states,
                time=np.array([0.0, 0.1, 0.2]), segment_distances=np.array([0.0, 2.0, 3.0]),
                captured=False, defense_failure=True, timeout=False,
                components={"near_shaping": near["original_near"], "original": np.array([-1.0, -102.0])})


CONFIG = dict(length_scale_m=20.0, beta=100.0, line_coefficient=0.1)


def test_replay_only_replaces_near_and_preserves_line_and_terminal_components():
    source = episode(moving=True)
    before = deepcopy(source)
    row, values = replay_episode(source, CONFIG, 0.99)
    assert row["original_return"] == -103.0
    assert row["line_return"] == pytest.approx(-0.05)
    assert row["new_with_line_return"] == pytest.approx(row["old_with_line_return"] - row["removed_near_return"])
    assert row["new_without_line_return"] == pytest.approx(-103.0 + row["active_near_return"])
    assert row["new_with_line_return_discounted"] == pytest.approx(
        row["old_with_line_return_discounted"] - row["removed_near_return_discounted"])
    np.testing.assert_array_equal(source["states"], before["states"])
    np.testing.assert_array_equal(source["components"]["near_shaping"], before["components"]["near_shaping"])
    assert values["row"] is row


def test_stationary_replay_eliminates_only_the_positive_near_component():
    row, _ = replay_episode(episode(cohort="stationary"), CONFIG, 0.99)
    assert row["old_near_return"] > 0
    assert row["active_near_return"] == 0
    assert row["new_with_line_return"] == pytest.approx(-103.05)
    assert row["removed_fraction"] == 1


def test_replay_rejects_old_shaping_mismatch():
    source = episode()
    source["components"]["near_shaping"][0] += 0.01
    with pytest.raises(AssertionError):
        replay_episode(source, CONFIG, 0.99)


def test_summary_deduplicates_categories_but_keeps_best_final_model_views():
    rows = []
    for cohort in ("test", "stationary"):
        row, _ = replay_episode(episode(cohort=cohort), CONFIG, 0.99)
        rows.append(row)
    final = dict(rows[0], model="final", duplicate_of="sample", episode_id="alias")
    rows.append(final)
    categories, _, stationary, seeds = summarize(rows)
    group = next(r for r in categories if r["cohort"] == "test")
    assert group["episodes"] == 1
    assert {r["model"] for r in seeds} == {"best", "final"}
    assert len(stationary) == 8
    assert all(r["near_count"] == 1 for r in stationary)


def test_existing_output_is_rejected_before_source_reads(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.offline_active_near_replay.ROOT", tmp_path)
    (tmp_path / "existing").mkdir()
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(dict(output_dir="existing")), encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(path)


def test_complete_analysis_writes_report_sources_and_representative_data(tmp_path, monkeypatch):
    reference = episode(moving=True)
    stationary = episode(cohort="stationary")
    stationary["episode_id"] = "stationary"
    current = dict(deepcopy(reference), dataset="current_k0p1")
    current_static = dict(deepcopy(stationary), dataset="current_k0p1")
    for source in (reference, stationary, current, current_static):
        source.update(distances=np.linalg.norm(source["states"][:, 1, :3] - source["states"][:, 0, :3], axis=1),
                      trajectory="synthetic_saved_fixture.npz", trajectory_sha256="fixture", content_sha256="fixture",
                      max_reward_reconstruction_error=0.0)
    environment = {1001: {"reward": {"best_so_far_shaping": {"beta": 100.0, "length_scale_m": 20.0}}}}
    monkeypatch.setattr("scripts.offline_active_near_replay.load_reference",
                        lambda config, hashes: ([reference, stationary], 0.99, environment, [{"episode_id": "sample"}]))
    monkeypatch.setattr("scripts.offline_active_near_replay.load_current",
                        lambda config, hashes, previous, gamma: ([current, current_static], environment))
    output = tmp_path / "replay"
    config = dict(CONFIG, output_dir=str(output), current_representative_categories=["near_miss"])
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = run(path)
    assert manifest["models_loaded"] is False
    assert manifest["new_inference_episodes"] == 0
    assert manifest["stationary_max_active_reward"] == 0
    assert manifest["source_files_unchanged"] is True
    for name in ("representatives.csv", "representative_step_components.csv", "verification.json",
                 "ACTIVE_NEAR_REPLAY_REPORT.md", "replay_manifest.json", "plots/old_vs_active_near.png"):
        assert (output / name).is_file()
    saved = json.loads((output / "replay_manifest.json").read_text(encoding="utf-8"))
    assert saved["unique_counts"]["reference_B/test"] == 1
    assert saved["reporting"]["report"] == "ACTIVE_NEAR_REPLAY_REPORT.md"
