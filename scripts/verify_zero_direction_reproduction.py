"""Reproduce v08's full-rollout schedule using v09's fixed-zero environment.

This independent validation entry point does not change either trainer. It
temporarily supplies DirectionCurriculumEnv(0 degrees) to the ordinary MAPPO
trainer, which collects full rollouts rather than shortening them for 10k
evaluations. Evaluation and curriculum promotion are deliberately outside this
experiment; this is a test of the proposed source of training divergence.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts import train_mappo
from utils.direction_curriculum import DirectionCurriculumEnv
from utils.experiment import ROOT, load_training_config


MODEL_KEYS = ("actor", "critic", "actor_optimizer", "critic_optimizer", "rng_state")
STATE_KEYS = (
    "env_steps", "update", "episodes", "partial_episode", "recent_episodes",
    "environment_state", "environment_accelerations", "environment_step_count",
    "environment_rng_state", "best_score",
)
LEARNING_KEYS = ("seed", "device", "torch_num_threads", "network", "ppo", "normalization")
TRAINING_KEYS = ("rollout_steps", "log_interval_updates", "checkpoint_interval_updates", "capture_rate_window")


class FixedZeroDirectionEnv(DirectionCurriculumEnv):
    """The actual v09 zero-width sampler, using the ordinary trainer interface."""

    def __init__(self, config_path):
        # No angle is drawn at zero width; the trainer passes its YAML seed to
        # reset(), which seeds this local sampler without touching global RNG.
        super().__init__(config_path, half_angle_degrees=0.0, sampling_seed=0)


@contextmanager
def fixed_zero_training_environment():
    """Restore the trainer's constructor even when a validation run fails.

The CLI is a single-process validation command; callers must not train another
agent concurrently inside the same Python interpreter while this is active.
"""
    original = train_mappo.PursuitEnv
    try:
        train_mappo.PursuitEnv = FixedZeroDirectionEnv
        yield
    finally:
        train_mappo.PursuitEnv = original


def _yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _resolve(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _steps(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in values):
        raise ValueError("reproduction.checkpoint_steps must be positive integer steps")
    return sorted(set(values))


def validate_reference(config, environment_path, reference_run=None):
    """Reject changed physical/learning settings before creating run output."""
    settings = config.get("reproduction", {})
    reference = _resolve(reference_run if reference_run is not None else settings["reference_run"])
    old_config = _yaml(reference / "training_config.yaml")
    old_environment = _yaml(reference / "environment_config.yaml")
    environment = _yaml(environment_path)
    if environment != old_environment:
        raise ValueError("Environment differs from the reference's actual snapshot")
    for key in LEARNING_KEYS:
        if config.get(key) != old_config.get(key):
            raise ValueError("Learning setting differs from reference: " + key)
    for key in TRAINING_KEYS:
        if config["training"].get(key) != old_config["training"].get(key):
            raise ValueError("Training setting differs from reference: " + key)
    if config.get("curriculum"):
        raise ValueError("This is a fixed-zero reproduction, not a curriculum run")
    if config.get("continuation") or config.get("initial_checkpoint") or config.get("resume_checkpoint"):
        raise ValueError("Reproduction must start from scratch")
    _steps(settings["checkpoint_steps"])
    if not (reference / "updates.csv").is_file() or not (reference / "training_metrics.csv").is_file():
        raise ValueError("Reference training logs are missing")
    return reference


def first_difference(actual, expected, path=""):
    """Return the first exact tensor/array/nested-state difference, or None."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or actual.shape != expected.shape:
            return path + ": tensor type/dtype/shape"
        return None if torch.equal(actual, expected) else path + ": tensor values"
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or actual.shape != expected.shape:
            return path + ": array type/dtype/shape"
        return None if np.array_equal(actual, expected) else path + ": array values"
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return path + ": dictionary keys"
        for key in expected:
            difference = first_difference(actual[key], expected[key], "{}[{}]".format(path, repr(key)))
            if difference is not None:
                return difference
        return None
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            return path + ": sequence type/length"
        for index, (left, right) in enumerate(zip(actual, expected)):
            difference = first_difference(left, right, "{}[{}]".format(path, index))
            if difference is not None:
                return difference
        return None
    return None if actual == expected else path + ": scalar value"


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def compare_csv(actual_path, reference_path, prefix_steps):
    """Compare every reference column in the common full-rollout prefix."""
    columns, actual = _rows(actual_path)
    expected_columns, expected = _rows(reference_path)
    missing = sorted(set(expected_columns) - set(columns))
    result = dict(file=Path(actual_path).name, prefix_steps=prefix_steps,
                  compared_columns=expected_columns,
                  extra_actual_columns=sorted(set(columns) - set(expected_columns)),
                  rows_compared=0, all_equal=False, first_difference=None)
    if missing:
        result["first_difference"] = dict(reason="Missing reference columns", columns=missing)
        return result
    actual = [row for row in actual if int(row["env_steps"]) <= prefix_steps]
    expected = [row for row in expected if int(row["env_steps"]) <= prefix_steps]
    for index, (left, right) in enumerate(zip(actual, expected)):
        differences = {key: dict(actual=left[key], expected=right[key])
                       for key in expected_columns if left[key] != right[key]}
        result["rows_compared"] = index + 1
        if differences:
            result["first_difference"] = dict(row=index + 1, env_steps=left["env_steps"], differences=differences)
            return result
    if len(actual) != len(expected):
        result["first_difference"] = dict(reason="Prefix row counts differ", actual=len(actual), expected=len(expected))
        return result
    result["all_equal"] = True
    return result


def compare_checkpoint(actual_path, reference_path):
    """Compare complete models/optimizers/RNG and required continuation state."""
    actual = torch.load(actual_path, map_location="cpu", weights_only=False)
    expected = torch.load(reference_path, map_location="cpu", weights_only=False)
    checks = {}
    for key in MODEL_KEYS:
        checks[key] = first_difference(actual.get(key), expected[key], key)
    for key in STATE_KEYS:
        label = "training_state." + key
        if key not in actual["training_state"] or key not in expected["training_state"]:
            checks[label] = label + ": missing required key"
        else:
            checks[label] = first_difference(actual["training_state"][key], expected["training_state"][key], label)
    # Metadata/config record different output paths, budgets and timestamps;
    # byte-identical .pt files are neither expected nor claimed.
    differences = {key: value for key, value in checks.items() if value is not None}
    return dict(checkpoint=Path(actual_path).name, reference_checkpoint=str(Path(reference_path).resolve()),
                env_steps=int(actual["training_state"]["env_steps"]),
                all_equal=not differences, differences=differences,
                actual_sha256=_sha(actual_path), reference_sha256=_sha(reference_path),
                checked_sections=list(MODEL_KEYS), checked_training_state_keys=list(STATE_KEYS),
                excluded_sections=["config", "metadata", "file_serialization"])


def compare_run(run_dir, reference_run, checkpoint_steps):
    """Read existing runs and write only the new run's comparison report."""
    run, reference = _resolve(run_dir), _resolve(reference_run)
    if run == reference:
        raise ValueError("Refusing to write a comparison into the historical reference run")
    config = _yaml(run / "training_config.yaml")
    requested = _steps(checkpoint_steps)
    final = torch.load(run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    total = int(final["training_state"]["env_steps"])
    rollout = config["training"]["rollout_steps"]
    prefix = total // rollout * rollout
    reference_configs = _yaml(reference / "training_config.yaml")
    if total > reference_configs["training"]["total_env_steps"]:
        raise ValueError("Reference is shorter than the reproduction")
    csv_results = [compare_csv(run / name, reference / name, prefix)
                   for name in ("updates.csv", "training_metrics.csv")]
    checkpoint_results = []
    for checkpoint in sorted((run / "checkpoints").glob("update_*.pt")):
        counterpart = reference / "checkpoints" / checkpoint.name
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["training_state"]["env_steps"] > prefix:
            continue
        if not counterpart.is_file():
            checkpoint_results.append(dict(checkpoint=checkpoint.name, all_equal=False,
                                           differences={"reference": "Missing reference checkpoint"}))
        else:
            checkpoint_results.append(compare_checkpoint(checkpoint, counterpart))
    requested_results = []
    for step in requested:
        status = "outside_completed_full_rollout_prefix" if step > prefix else "missing"
        match = next((row for row in checkpoint_results if row.get("env_steps") == step), None)
        if match is not None:
            status = "equal" if match["all_equal"] else "different"
        requested_results.append(dict(step=step, status=status,
                                      checkpoint=None if match is None else match["checkpoint"]))
    all_equal = bool(checkpoint_results) and all(row["all_equal"] for row in csv_results + checkpoint_results)
    all_equal = all_equal and all(row["status"] in ("equal", "outside_completed_full_rollout_prefix") for row in requested_results)
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(run),
        reference_run=str(reference), actual_steps=total, rollout_steps=rollout,
        compared_full_rollout_prefix_steps=prefix,
        excluded_final_short_rollout_steps=total - prefix,
        environment_class="DirectionCurriculumEnv", half_angle_degrees=0.0,
        schedule="Unchanged scripts.train_mappo full-rollout updates; no 10k truncation",
        all_comparable_state_equal=all_equal, csv_comparisons=csv_results,
        checkpoint_comparisons=checkpoint_results, requested_checkpoints=requested_results,
        caveats=["This is a schedule-reproduction check, not a curriculum promotion run.",
                 "The reference's missing initial checkpoint is not reconstructed.",
                 "The final short rollout has a different update horizon from the longer reference run and is excluded from prefix equivalence.",
                 "Checkpoint equality means exact state tensors/arrays, not identical metadata or .pt file bytes."])
    (run / "reproduction_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def run_verification(config_path, *, total_steps=None, output_dir=None, compare_only=False):
    config, environment = load_training_config(config_path)
    reference = validate_reference(config, environment)
    destination = _resolve(output_dir if output_dir is not None else config["output_dir"])
    # The trainer also refuses preexisting artifacts. This explicit check
    # protects the reference even if a caller supplies its path accidentally.
    if destination == reference or reference in destination.parents:
        raise ValueError("Reproduction output must be independent of the reference run")
    if not compare_only:
        with fixed_zero_training_environment():
            train_mappo.train(config_path, total_steps=total_steps, output_dir=destination)
    result = compare_run(destination, reference, config["reproduction"]["checkpoint_steps"])
    if not result["all_comparable_state_equal"]:
        raise RuntimeError("Reproduction diverged; inspect " + str(destination / "reproduction_comparison.json"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_verification(args.config, total_steps=args.total_steps,
                              output_dir=args.output_dir, compare_only=args.compare_only)
    print(json.dumps({key: result[key] for key in ("run_dir", "actual_steps", "compared_full_rollout_prefix_steps",
        "excluded_final_short_rollout_steps", "all_comparable_state_equal", "requested_checkpoints")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
