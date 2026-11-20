"""Record and verify the last curriculum stage's best and final full videos.

Both checkpoints replay one identical, already evaluated fixed-grid scene.
The scene is the middle successful best-eval angle, or the middle grid angle
when best-eval has no successes. This illustration is not a capture-rate test.
Existing video directories and original experiment artifacts are preserved.
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.mappo import MAPPOAgent
from scripts.render_mappo_video import render_episode
from utils.direction_curriculum import DirectionCurriculumEnv
from utils.evaluation import run_episode, save_trajectory
from utils.experiment import ROOT, set_seed


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    _require(bool(records), "Evaluation CSV is empty: " + str(path))
    return records


def _assert_metrics(actual, expected):
    """Compare every rerun metric against the original evaluation CSV row."""
    checks = {}
    for name, value in actual.items():
        _require(name in expected, "Reference CSV has no metric: " + name)
        reference = expected[name]
        if value is None:
            matched = reference == ""
        elif isinstance(value, bool):
            matched = reference == str(value)
        elif isinstance(value, str):
            matched = value == reference
        elif isinstance(value, (int, np.integer)):
            matched = value == int(reference)
        else:
            matched = bool(np.isfinite(value) and np.isclose(value, float(reference), rtol=1e-10, atol=1e-10))
        _require(matched, "Rerun metric differs from evaluation: {} ({} vs {})".format(name, value, reference))
        checks[name] = dict(actual=value, reference_csv=reference, matched=True)
    return checks


def _checkpoint_record(run, stage_index, half_angle, role, filename):
    stage = run / "stage_{:02d}_pm{:g}".format(stage_index, half_angle)
    checkpoint = stage / "checkpoints" / filename
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["training_state"]
    _require(state["stage_index"] == stage_index, "Checkpoint belongs to a different curriculum stage")
    _require(float(state["stage_half_angle_degrees"]) == half_angle, "Checkpoint angle range differs from stage")
    warm_initial = (state["stage_env_steps"] == 0
                    and state.get("source_pretrained_env_steps", 0) > 0)
    _require((state["stage_env_steps"] > 0 or warm_initial)
             and state["env_steps"] >= state["stage_env_steps"],
             "Invalid checkpoint step counters")
    evaluation = stage / "evaluations" / "at_{:06d}".format(state["stage_env_steps"]) / "current"
    csv_path = evaluation / "evaluation_metrics.csv"
    group = _json(evaluation / "summary.json")
    return dict(role=role, checkpoint=checkpoint.resolve(), payload=payload, stage=stage,
                evaluation=evaluation, csv_path=csv_path, rows=_rows(csv_path), group=group)


def record_videos(run_dir):
    """Replay two policies, preserve source hashes, then render and decode QA."""
    run = Path(run_dir).expanduser()
    run = (run if run.is_absolute() else ROOT / run).resolve()
    summary_path = run / "summary.json"
    summary = _json(summary_path)
    _require(summary.get("status") in ("completed", "stopped_stage_budget", "stopped_total_budget"),
             "Videos require a completed or budget-stopped curriculum run")
    _require(bool(summary.get("stages")), "Run summary contains no curriculum stage")
    output = run / "videos"
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing video directory: " + str(output))
    environment_path = run / "environment_config.yaml"
    training_path = run / "training_config.yaml"
    environment_config = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    training_config = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    last_stage = summary["stages"][-1]
    stage_index = int(last_stage["stage_index"])
    half_angle = float(last_stage["half_angle_degrees"])
    records = [
        _checkpoint_record(run, stage_index, half_angle, "best", "best_eval.pt"),
        _checkpoint_record(run, stage_index, half_angle, "final", "last.pt"),
    ]
    for record in records:
        _require(record["payload"]["metadata"]["environment_config"] == environment_config,
                 "Checkpoint environment snapshot differs from this run: " + record["role"])
        _require(len(record["rows"]) == record["group"]["episodes"], "Incomplete reference evaluation CSV")
    _require(records[1]["payload"]["training_state"]["env_steps"] == summary["env_steps"],
             "last.pt is not the terminal checkpoint identified by the run summary")
    successes = [row for row in records[0]["rows"] if row["captured"] == "True"]
    candidates = successes if successes else records[0]["rows"]
    selected = candidates[len(candidates) // 2]
    angle, seed = float(selected["angle_degrees"]), int(selected["seed"])
    selection = (
        "Middle of best-eval's {} successful fixed-grid angles; both policies use this identical angle and seed. "
        "This illustrative success is not a representative capture-rate estimate."
    ).format(len(successes)) if successes else (
        "Best-eval has no successful fixed-grid angles; use the middle of its {} evaluation angles for both "
        "policies, with an identical seed. This illustration is not a capture-rate estimate."
    ).format(len(candidates))
    for record in records:
        matches = [row for row in record["rows"]
                   if int(row["seed"]) == seed and np.isclose(float(row["angle_degrees"]), angle, rtol=0, atol=1e-12)]
        _require(len(matches) == 1, "Expected exactly one matching angle/seed in " + str(record["csv_path"]))
        record["reference_row"] = matches[0]

    protected = {summary_path, environment_path, training_path}
    for relative in ("curriculum_state.json", "evaluation_history.csv", "metadata.json"):
        candidate = run / relative
        if candidate.is_file():
            protected.add(candidate)
    # Include old and new stage metrics, not only the two selected CSV rows.
    protected.update(run.glob("stage_*/**/*.csv"))
    for record in records:
        protected.update((record["checkpoint"], record["csv_path"], record["evaluation"] / "summary.json"))
    hashes_before = {str(path.resolve()): _sha(path) for path in sorted(protected)}
    output.mkdir(parents=False, exist_ok=False)
    _write_json(output / "source_integrity_before.json", hashes_before)
    models, manifests = [], []
    first_initial_state = None
    for record in records:
        role, checkpoint = record["role"], record["checkpoint"]
        directory = output / role
        directory.mkdir()
        set_seed(seed, int(training_config["torch_num_threads"]))
        agent, payload = MAPPOAgent.from_checkpoint(checkpoint, device=training_config["device"])
        env = DirectionCurriculumEnv(environment_path, half_angle, seed, fixed_angle_degrees=angle)
        with torch.inference_mode():
            metrics, trajectory = run_episode(env, agent, seed)
        for name, array in trajectory.items():
            if np.issubdtype(array.dtype, np.number):
                _require(np.isfinite(array).all(), "Non-finite recorded trajectory: " + name)
        checks = _assert_metrics(metrics, record["reference_row"])
        if first_initial_state is None:
            first_initial_state = trajectory["states"][0].copy()
        else:
            _require(np.array_equal(first_initial_state, trajectory["states"][0]),
                     "Best and final initial states differ")
        for index, coordinate in enumerate(("x", "y", "z")):
            _require(np.isclose(trajectory["positions"][0, -1, index],
                                float(record["reference_row"]["initial_attacker_" + coordinate]),
                                rtol=0, atol=1e-10), "Initial attacker position differs from reference CSV")
        actual_config = deepcopy(env.config)
        actual_config["initialization"]["attacker_position"] = trajectory["positions"][0, -1].tolist()
        actual_config_path = directory / "environment_config.yaml"
        with actual_config_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(actual_config, stream, sort_keys=False, allow_unicode=True)
        trajectory_path = directory / "trajectory.npz"
        save_trajectory(trajectory_path, trajectory, metrics, seed)
        metadata = dict(
            created_at_utc=datetime.now(timezone.utc).isoformat(), model=role,
            label="V09 BEST-EVAL" if role == "best" else "V09 FINAL (last.pt)",
            checkpoint=str(checkpoint), checkpoint_sha256=hashes_before[str(checkpoint)],
            checkpoint_training_state={key: payload["training_state"][key]
                                       for key in ("env_steps", "update", "episodes", "stage_index", "stage_env_steps")},
            source_sha256=_sha(trajectory_path), config_sha256=_sha(actual_config_path),
            source_environment_config=str(environment_path), source_environment_config_sha256=_sha(environment_path),
            angle_degrees=angle, seed=seed, deterministic=True,
            initial_attacker_position=trajectory["positions"][0, -1].tolist(), selection=selection,
            reference_evaluation=str(record["evaluation"].resolve()),
            reference_episode=int(record["reference_row"]["episode"]),
            reference_group_capture_rate=record["group"]["capture_rate"], simulation_rerun=True,
            metrics_match_existing_evaluation=True, episode_metrics=metrics,
        )
        if payload["training_state"].get("source_pretrained_env_steps", 0):
            metadata["source_pretrained_env_steps"] = payload["training_state"]["source_pretrained_env_steps"]
            metadata["step_count_semantics"] = "Checkpoint steps count newly collected curriculum work; source pretraining is separate."
        _write_json(directory / "episode_metadata.json", metadata)
        _write_json(directory / "reference_evaluation_row.json", record["reference_row"])
        _write_json(directory / "metrics_verification.json", dict(all_checks_passed=True, fields=checks))
        print(json.dumps(dict(model=role, angle_degrees=angle, seed=seed, metrics=metrics), ensure_ascii=False), flush=True)
        models.append(metadata)
        del agent, payload, env, trajectory
        manifests.append(render_episode(directory))

    hashes_after = {name: _sha(Path(name)) for name in hashes_before}
    _require(hashes_after == hashes_before, "Original experiment sources changed while recording/rendering")
    verification = []
    for model, manifest in zip(models, manifests):
        directory = output / model["model"]
        checked = _json(directory / "video_verification.json")
        _require(checked["all_checks_passed"], "Renderer verification failed for " + model["model"])
        video = directory / (model["model"] + "_full_episode.mp4")
        _require(_sha(video) == manifest["video_sha256"], "Delivered video SHA differs from renderer manifest")
        verification.append(dict(model=model["model"], video=str(video), video_sha256=manifest["video_sha256"],
                                 all_checks_passed=True, decoded_frames=checked["decoded_frames"],
                                 includes_initial_and_final_states=True,
                                 metrics_match_existing_evaluation=True))
    comparison = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(run),
        stage_index=stage_index, half_angle_degrees=half_angle, angle_degrees=angle, seed=seed,
        same_initial_conditions=True, scene_selection=selection, models=models,
        protected_source_sha256_before=hashes_before, protected_source_sha256_after=hashes_after,
        all_original_sources_unchanged=True,
    )
    _write_json(output / "comparison_summary.json", comparison)
    _write_json(output / "video_delivery_verification.json", dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), all_checks_passed=True,
        all_original_sources_unchanged=True, same_initial_conditions=True, videos=verification,
        protected_source_sha256_before=hashes_before, protected_source_sha256_after=hashes_after,
    ))
    print("Saved two fully verified videos to " + str(output), flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    record_videos(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
