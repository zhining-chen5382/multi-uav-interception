"""Reporting acceptance with tiny engineering fixtures and real MP4 decoding.

Normalization divisors here are test fixtures. They do not establish the
missing user-confirmed production observation contract.
"""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from utils import formal_ppo_reporting as reporting
from utils import formal_ppo_training as training


ROOT = Path(__file__).resolve().parents[1]
TEST_SCALES = [20, 20, 8, 3, 3, 2, 1500, 1500, 1500, 40, 40, 14, 750, 750, 750, 1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reporting")
    config = training.load_configuration(ROOT / "configs/ppo_formal_v1.yaml")
    config["environment_config"]["normalization"].update(
        confirmed=True, scales=TEST_SCALES, acceleration_kind="applied",
        confirmation_source="Reporting unit-test fixture only")
    config["environment_config"]["environment"]["max_episode_steps"] = 2
    config["training"].update(total_env_steps=16, num_envs=2, rollout_steps=2,
                              evaluation_interval_steps=4, validation_episodes=2)
    config["network"].update(actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8])
    config["ppo"].update(update_epochs=1, minibatch_size=4, target_kl=1000.)
    run = directory / "tiny_completed_run"
    summary = training.train(config, run, 1001)
    # A misleading filename must never become the reported training step.
    shutil.copyfile(run / "checkpoints/initial.pt", run / "checkpoints/checkpoint_999999.pt")
    source_paths = list(run.glob("*.csv")) + list((run / "checkpoints").glob("*.pt"))
    before = {str(path): sha(path) for path in source_paths}
    rng_before = training._state_hash(training.global_rng_state())
    manifest = reporting.postprocess_run(run, config, summary)
    rng_after = training._state_hash(training.global_rng_state())
    return dict(run=run, config=config, summary=summary, manifest=manifest,
                output=Path(manifest["output_dir"]), before=before,
                rng_before=rng_before, rng_after=rng_after)


def test_complete_callback_pipeline_and_source_preservation(completed):
    manifest, output = completed["manifest"], completed["output"]
    assert manifest["status"] == "completed"
    assert output.name == "postprocessing_001"
    assert manifest["source_artifacts_unchanged"] is True
    assert completed["rng_before"] == completed["rng_after"]
    for path, digest in completed["before"].items():
        assert sha(path) == digest
    assert read_json(output / "postprocessing_manifest.json") == manifest
    assert (output / "resolved_config.yaml").is_file()
    report = (output / "TRAINING_REPORT.md").read_text(encoding="utf-8")
    assert "16" in report
    assert "Reporting unit-test fixture only" in report
    assert "未完成 rollout" in report
    assert "未进行滑动平均" in report
    navigation = (completed["run"] / "README.md").read_text(encoding="utf-8")
    assert "postprocessing_001/TRAINING_REPORT.md" in navigation
    assert "postprocessing_001/videos/best/best_full_episode.mp4" in navigation
    assert "[完整验证场景与轨迹](validation/)" in navigation
    assert (output / "README.md").is_file()


def test_every_real_checkpoint_uses_saved_steps_and_rollout_metadata(completed):
    run, output = completed["run"], completed["output"]
    rows = read_csv(output / "trajectory_evolution/checkpoint_metrics.csv")
    paths = list((run / "checkpoints").glob("*.pt"))
    assert len(rows) == len(paths) == completed["manifest"]["checkpoint_count"]
    assert [int(row["step"]) for row in rows] == sorted(int(row["step"]) for row in rows)
    for row in rows:
        checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
        state = checkpoint["training_state"]
        assert int(row["step"]) == state["env_steps"]
        assert int(row["last_update_env_steps"]) == state["last_update_env_steps"]
        assert int(row["pending_rollout_samples"]) == state["rollout_buffer"]["size"] * 2
        assert float(row["angle_degrees"]) == 0
        assert int(row["episode_length"]) == 2
        assert row["result"] == "Timeout"
    named = next(row for row in rows if Path(row["checkpoint"]).name == "checkpoint_999999.pt")
    assert named["step"] == "0"
    assert any(row["inference_reused_from"] for row in rows)
    sampling = read_json(output / "trajectory_evolution/sampling_manifest.json")
    assert sampling["checkpoints"] > sampling["distinct_actor_policies"]
    assert sampling["missing_checkpoints_recreated"] is False


def test_all_requested_raw_metric_plots_exist_and_are_nonblank(completed):
    output, plots = completed["output"], completed["manifest"]["plots"]
    expected = {"capture_and_reward.png", "actor_critic_loss.png", "learning_rates.png",
                "log_std_and_std.png", "entropy_kl_clip_fraction.png",
                "episode_length_and_min_distance.png", "minibatch_analytic_kl.png"}
    assert expected == {Path(path).name for path in plots}
    paths = [output / path for path in plots]
    paths.append(output / "trajectory_evolution/trajectory_evolution_xy.png")
    for path in paths:
        with Image.open(path) as image:
            assert image.width >= 900
            assert np.asarray(image.convert("RGB")).std() > 5
    assert completed["manifest"]["smoothing"] == "None; raw CSV values only"


@pytest.mark.parametrize("model", ["best", "final"])
def test_complete_two_aircraft_video_manifest_and_full_decode(completed, model):
    output = completed["output"]
    directory = output / "videos" / model
    manifest = read_json(directory / (model + "_video_manifest.json"))
    verification = read_json(directory / "decode_verification.json")
    assert manifest["checkpoint_env_steps"] >= 0
    assert manifest["angle_degrees"] == 0
    assert manifest["episode_steps"] == 2
    assert manifest["frames"] == 3
    assert manifest["simulation_seconds"] == pytest.approx(.2)
    assert manifest["video_seconds"] == pytest.approx(.3)
    assert sha(manifest["video"]) == manifest["video_sha256"]
    assert sha(manifest["trajectory"]) == manifest["trajectory_sha256"]
    assert verification["all_checks_passed"]
    assert verification["decoded_every_frame"]
    assert verification["decoded_frames"] == 3
    assert verification["includes_initial_state"]
    assert verification["includes_terminal_state"]
    assert set(verification["sampled_frames"]) == {"0", "1", "2"}
    with np.load(manifest["trajectory"], allow_pickle=False) as data:
        assert data["states"].shape == (3, 2, 6)
        assert data["actions"].shape == (2, 3)
        np.testing.assert_array_equal(data["time"], [0, .1, .2])


def test_best_and_final_share_exact_initial_scene(completed):
    manifests = completed["manifest"]["videos"]
    with np.load(manifests["best"]["trajectory"], allow_pickle=False) as best:
        with np.load(manifests["final"]["trajectory"], allow_pickle=False) as final:
            np.testing.assert_array_equal(best["states"][0], final["states"][0])
    assert manifests["best"]["reference_validation_capture_rate"] == 0


def test_repeated_postprocessing_writes_independent_directory(completed):
    first = completed["output"]
    original_manifest_hash = sha(first / "postprocessing_manifest.json")
    original_navigation_hash = sha(completed["run"] / "README.md")
    manifest = reporting.postprocess_run(completed["run"])
    assert Path(manifest["output_dir"]).name == "postprocessing_002"
    assert sha(first / "postprocessing_manifest.json") == original_manifest_hash
    assert sha(completed["run"] / "README.md") == original_navigation_hash
    assert Path(manifest["output_dir"], "videos/final/final_full_episode.mp4").is_file()


def test_configuration_mismatch_is_rejected_before_writing(completed):
    config = deepcopy(completed["config"])
    config["ppo"]["entropy_coefficient"] = .123
    existing = set(completed["run"].glob("postprocessing_*"))
    with pytest.raises(ValueError, match="config differs"):
        reporting.postprocess_run(completed["run"], config, completed["summary"])
    assert set(completed["run"].glob("postprocessing_*")) == existing


def test_production_normalization_gate_survives_reporting(completed, tmp_path):
    run = tmp_path / "blocked"
    (run / "checkpoints").mkdir(parents=True)
    bundle = torch.load(completed["run"] / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    bundle["environment_config"]["normalization"].update(confirmed=False, scales=None, acceleration_kind=None)
    for name in ("best", "final"):
        torch.save(bundle, run / "checkpoints" / (name + ".pt"))
    with pytest.raises(ValueError, match="confirm"):
        reporting.postprocess_run(run)
    assert not list(run.glob("postprocessing_*"))


def test_partial_episode_rejected(completed, tmp_path):
    source = Path(completed["manifest"]["videos"]["final"]["trajectory"])
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["states"] = arrays["states"][:-1]
    arrays["time"] = arrays["time"][:-1]
    target = tmp_path / "incomplete.npz"
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError, match="Incomplete"):
        reporting._read_trajectory(target)


def test_no_checkpoints_or_missing_best_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="No real saved checkpoints"):
        reporting.postprocess_run(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/final.pt").write_bytes(b"Not a checkpoint")
    with pytest.raises(ValueError, match="both final.pt and best.pt"):
        reporting.postprocess_run(tmp_path)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_source_metrics_fail_explicitly(value):
    with pytest.raises(FloatingPointError, match="Non-finite training metric"):
        reporting._numbers([dict(actor_loss=value)], "actor_loss")
