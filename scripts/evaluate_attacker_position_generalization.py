"""Evaluate a frozen MAPPO checkpoint with independent XYZ attacker offsets.

Only per-episode copies of initialization.attacker_position change. The training
code, environment implementation and source checkpoint are never modified.
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import yaml

from algorithms.mappo import MAPPOAgent
from envs import PursuitEnv
from utils.evaluation import (
    checkpoint_provenance, run_episode, save_trajectory, summarize_episodes, write_json,
)
from utils.experiment import ROOT, set_seed, write_metadata


DEFAULT_CONFIG = ROOT / "configs/eval_attacker_position_generalization.yaml"


def _position(value):
    if np.iscomplexobj(value):
        raise ValueError("Position must contain three real finite coordinates")
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Position must contain three real finite coordinates")
    return position


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def sample_initial_position(center, margin, seed):
    """Return center + three independent uniform offsets, with no projection."""
    center = _position(center)
    margin = float(margin)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("The position half-width must be finite and nonnegative")
    _nonnegative_integer(seed, "seed")
    return center + np.random.default_rng(seed).uniform(-margin, margin, size=3)


def episode_environment_config(base, position):
    """Copy an environment configuration, replacing only attacker position."""
    result = deepcopy(base)
    result["initialization"]["attacker_position"] = _position(position).tolist()
    return result


def validate_protocol(config):
    """Reject ambiguous seeds, invalid offsets and unsafe group directories."""
    if not isinstance(config, dict):
        raise ValueError("Evaluation configuration must be a mapping")
    episodes = _nonnegative_integer(config["episodes_per_group"], "episodes_per_group")
    stride = _nonnegative_integer(config["group_seed_stride"], "group_seed_stride")
    _nonnegative_integer(config["base_seed"], "base_seed")
    if episodes == 0 or stride < episodes:
        raise ValueError("Require positive episodes and disjoint group seed ranges")
    groups = config["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list")
    names, margins = [], []
    for group in groups:
        name = group["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Group names may contain only letters, digits, '_' and '-'")
        margin = float(group["half_width_m"])
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Group half-widths must be finite and nonnegative")
        names.append(name)
        margins.append(margin)
    if len(set(names)) != len(names) or margins.count(0.0) != 1:
        raise ValueError("Require unique group names and exactly one fixed group")
    if margins[0] != 0.0:
        raise ValueError("The first group must be the fixed-state reference")
    expected_sha = config["expected_checkpoint_sha256"]
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("expected_checkpoint_sha256 must be a SHA256 digest")
    if _nonnegative_integer(config["expected_env_steps"], "expected_env_steps") == 0:
        raise ValueError("expected_env_steps must be positive")


def _path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_sha(value):
    """Hash tensor values, shapes and types without serialization timestamps."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode("ascii"))
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"mapping")
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                add(child)
        else:
            digest.update((type(item).__name__ + ":" + repr(item)).encode("utf-8"))

    add(value)
    return digest.hexdigest()


def _frozen_hashes(agent):
    return {
        "actor": _state_sha(agent.actor.state_dict()),
        "critic": _state_sha(agent.critic.state_dict()),
        "actor_optimizer": _state_sha(agent.actor_optimizer.state_dict()),
        "critic_optimizer": _state_sha(agent.critic_optimizer.state_dict()),
        "torch_rng": _state_sha(torch.get_rng_state()),
    }


def _write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config_path=DEFAULT_CONFIG, output_dir=None):
    """Run the configured fixed/randomized groups once, without any updates."""
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    checkpoint = _path(config["checkpoint"])
    checkpoint_sha = _sha(checkpoint)
    if checkpoint_sha != config["expected_checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from the confirmed model")
    output = _path(config["output_dir"] if output_dir is None else output_dir)
    output.relative_to(ROOT)
    artifacts = ("metadata.json", "summary.json", "episode_metrics.csv", "groups",
                 "evaluation_protocol.yaml", "baseline_environment_config.yaml")
    if output.exists() and (not output.is_dir() or any((output / name).exists() for name in artifacts)):
        raise ValueError("Evaluation artifacts already exist; refusing to overwrite: " + str(output))

    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    agent, payload = MAPPOAgent.from_checkpoint(checkpoint, device="cpu")
    if payload["training_state"]["env_steps"] != config["expected_env_steps"]:
        raise ValueError("Checkpoint training steps differ from the confirmed model")
    base = deepcopy(payload["metadata"]["environment_config"])
    reference_config = _path(config["reference_environment_config"])
    if base != yaml.safe_load(reference_config.read_text(encoding="utf-8")):
        raise ValueError("Confirmed environment YAML differs from checkpoint snapshot")
    if base.get("attacker", {}).get("policy", {}).get("mode") != "direct":
        raise ValueError("This protocol requires the unchanged Direct Attacker")
    center = _position(base["initialization"]["attacker_position"])
    bounds = base["environment"]["bounds"]
    largest_margin = max(float(group["half_width_m"]) for group in config["groups"])
    if np.any(center - largest_margin < bounds["low"]) or np.any(center + largest_margin > bounds["high"]):
        raise ValueError("Requested sampling cube exceeds workspace; no clipping or resampling allowed")

    set_seed(config["base_seed"], payload["config"]["torch_num_threads"])
    for network in (agent.actor, agent.critic):
        network.eval()
        network.requires_grad_(False)
    before = _frozen_hashes(agent)
    output.mkdir(parents=True, exist_ok=True)
    for name, mapping in (("evaluation_protocol.yaml", config), ("baseline_environment_config.yaml", base)):
        (output / name).write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
    reference_trajectory = _path(config["reference_trajectory"])
    reference_sha = _sha(reference_trajectory)
    rows, summaries = [], {}
    reference_arrays, fixed_fingerprint = None, None
    expected_pursuer_positions = np.asarray(base["initialization"]["pursuer_positions"])
    expected_velocities = np.vstack((base["initialization"]["pursuer_velocities"],
                                     base["initialization"]["attacker_velocity"]))

    with torch.inference_mode():
        for group_index, group in enumerate(config["groups"]):
            name, margin = group["name"], float(group["half_width_m"])
            folder = output / "groups" / name
            (folder / "environment_configs").mkdir(parents=True)
            (folder / "trajectories").mkdir()
            group_rows = []
            for episode_index in range(config["episodes_per_group"]):
                seed = config["base_seed"] + group_index * config["group_seed_stride"] + episode_index
                position = sample_initial_position(center, margin, seed)
                scenario = episode_environment_config(base, position)
                scenario_path = folder / "environment_configs" / "episode_{:03d}.yaml".format(episode_index + 1)
                scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True), encoding="utf-8")
                env = PursuitEnv(scenario_path)
                env.reset(seed=seed)
                np.testing.assert_array_equal(env.states[:3, :3], expected_pursuer_positions)
                np.testing.assert_array_equal(env.states[-1, :3], position)
                np.testing.assert_array_equal(env.states[:, 3:], expected_velocities)
                np.testing.assert_array_equal(env.accelerations, np.zeros_like(env.accelerations))
                metrics, trajectory = run_episode(env, agent, seed)
                for value in trajectory.values():
                    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                        raise RuntimeError("Non-finite trajectory in " + name)
                np.testing.assert_array_equal(trajectory["positions"][0, :3], expected_pursuer_positions)
                np.testing.assert_array_equal(trajectory["positions"][0, -1], position)
                np.testing.assert_array_equal(trajectory["states"][0, :, 3:], expected_velocities)
                if margin == 0.0:
                    fingerprint = _state_sha(trajectory)
                    if episode_index == 0:
                        with np.load(reference_trajectory, allow_pickle=False) as reference:
                            reference_arrays = sorted(set(reference.files) - {"metadata_json"})
                            if set(reference_arrays) != set(trajectory):
                                raise RuntimeError("Fixed reference array schema differs")
                            for key in reference_arrays:
                                np.testing.assert_array_equal(trajectory[key], reference[key])
                        fixed_fingerprint = fingerprint
                    elif fingerprint != fixed_fingerprint:
                        raise RuntimeError("Repeated fixed-state deterministic trajectories differ")
                trajectory_path = folder / "trajectories" / "episode_{:03d}.npz".format(episode_index + 1)
                save_trajectory(trajectory_path, trajectory, metrics, seed)
                offset = position - center
                row = {
                    "group": name, "half_width_m": margin, "episode": episode_index + 1,
                    "seed": seed, "initial_attacker_x": float(position[0]),
                    "initial_attacker_y": float(position[1]), "initial_attacker_z": float(position[2]),
                    "offset_x": float(offset[0]), "offset_y": float(offset[1]), "offset_z": float(offset[2]),
                    "environment_config": _relative(scenario_path), "trajectory": _relative(trajectory_path),
                    **metrics,
                }
                group_rows.append(row)
                rows.append(row)
                if (episode_index + 1) % 25 == 0:
                    print("{}: {}/{} episodes, captured={}".format(
                        name, episode_index + 1, config["episodes_per_group"],
                        sum(entry["captured"] for entry in group_rows)), flush=True)
            _write_csv(folder / "episode_metrics.csv", group_rows)
            summary = dict(summarize_episodes(group_rows), group=name, half_width_m=margin,
                           episode_seed_range=[group_rows[0]["seed"], group_rows[-1]["seed"]],
                           unique_initial_positions=len(set((r["initial_attacker_x"], r["initial_attacker_y"], r["initial_attacker_z"]) for r in group_rows)),
                           average_collision_steps=float(np.mean([r["collision_steps"] for r in group_rows])))
            summaries[name] = summary
            write_json(folder / "summary.json", summary)

    after = _frozen_hashes(agent)
    if before != after or _sha(checkpoint) != checkpoint_sha or _sha(reference_trajectory) != reference_sha:
        raise RuntimeError("Frozen network, optimizer, RNG or source artifacts changed")
    verification = {key + "_sha256_" + suffix: digest
                    for suffix, mapping in (("before", before), ("after", after))
                    for key, digest in mapping.items()}
    verification.update(
        no_training_updates=True, all_parameters_require_grad_false=all(
            not parameter.requires_grad for network in (agent.actor, agent.critic) for parameter in network.parameters()),
        actor_eval_mode=not agent.actor.training, critic_eval_mode=not agent.critic.training,
        all_episode_initial_accelerations_zero=True, all_episode_initial_states_match_episode_config=True,
        checkpoint_sha256_unchanged=True, fixed_reference_array_names=reference_arrays,
        fixed_first_episode_arrays_bit_identical=True, fixed_episodes_identical_ignoring_seed=True,
    )
    _write_csv(output / "episode_metrics.csv", rows)
    training_config = deepcopy(payload["config"])
    training_config["seed"] = config["base_seed"]
    metadata = write_metadata(
        output / "metadata.json", training_config, base, config_path=config_path,
        env_config_path=reference_config, output_dir=output, kind="generalization_evaluation",
        extra={"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
               "checkpoint_training_state": checkpoint_provenance(payload),
               "deterministic": True, "no_training": True, "environment_source": "checkpoint:metadata.environment_config",
               "only_environment_change": "initialization.attacker_position",
               "evaluation_protocol": config, "frozen_state_verification": verification,
               "reference_trajectory": str(reference_trajectory), "reference_trajectory_sha256": reference_sha,
               "source_sha256": {_relative(Path(__file__)): _sha(Path(__file__))},
               "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": time.perf_counter() - start_clock},
    )
    summary = {
        "episodes_total": len(rows), "groups": summaries,
        "checkpoint": _relative(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_training_steps": payload["training_state"]["env_steps"],
        "sampling": "Independent XYZ uniform offsets; disjoint configured group seed ranges; no spherical projection, rejection sampling or clipping",
        "collision_definitions": {"collision_episode_rate": "Fraction of episodes with at least one unsafe pursuer pair step",
                                  "collision_step_rate": "Total unsafe steps divided by total episode steps; team step counted once"},
        "fixed_group_note": "One fixed initial state repeated deterministically; not 100 independent generalization scenarios",
        "frozen_state_verification": verification,
        "completed_at_utc": metadata["completed_at_utc"],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"episodes_total": len(rows), "capture_rates": {name: value["capture_rate"] for name, value in summaries.items()},
                      "output_dir": str(output), "frozen_verified": True}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluate(args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
