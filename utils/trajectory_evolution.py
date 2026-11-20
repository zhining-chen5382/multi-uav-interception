"""Frozen checkpoint trajectory comparisons in one reproducible test scene.

This module only reads saved models. It never updates a policy, changes a
training checkpoint, or reconstructs an unavailable initial policy.
"""

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from algorithms.mappo import MAPPOAgent
from envs import PursuitEnv
from utils.direction_curriculum import arrival_position, restore_rng_state
from utils.evaluation import run_episode, save_trajectory, write_json
from utils.experiment import ROOT, set_seed


DEFAULT_CONFIG = ROOT / "configs/trajectory_evolution.yaml"
METRIC_FIELDS = (
    "step", "result", "captured", "episode_length", "event_time",
    "min_pursuer_attacker_distance", "final_attacker_base_distance",
    "episode_reward", "checkpoint", "checkpoint_sha256", "trajectory",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _steps(values, name):
    _require(isinstance(values, (list, tuple)), name + " must be a list of nonnegative integer steps")
    _require(all(not isinstance(value, bool) and isinstance(value, (int, np.integer))
                 and value >= 0 for value in values), name + " must contain nonnegative integer steps")
    return [int(value) for value in values]


def _real(value, name):
    _require(not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
             and not np.iscomplexobj(value), name + " must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " must be a finite real scalar") from error
    _require(np.isfinite(result), name + " must be finite")
    return result


def load_evolution_config(config_path=None):
    """Load plotting/evaluation settings from YAML, with explicit overrides."""
    with DEFAULT_CONFIG.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    _require(isinstance(config, dict), "Trajectory evolution configuration must be a mapping")
    if config_path is not None and _resolve(config_path) != DEFAULT_CONFIG.resolve():
        with _resolve(config_path).open(encoding="utf-8") as stream:
            override = yaml.safe_load(stream)
        _require(isinstance(override, dict), "Trajectory evolution overrides must be a mapping")
        config.update(override)
    _require(isinstance(config["enabled"], bool), "enabled must be boolean")
    config["target_steps"] = _steps(config["target_steps"], "target_steps")
    config["extra_checkpoint_steps"] = _steps(config.get("extra_checkpoint_steps", []), "extra_checkpoint_steps")
    _require(not isinstance(config["seed"], bool) and isinstance(config["seed"], int)
             and config["seed"] >= 0, "seed must be a nonnegative integer")
    config["angle_degrees"] = _real(config["angle_degrees"], "angle_degrees")
    _require(isinstance(config["device"], str) and config["device"], "device must name a PyTorch device")
    _require(not isinstance(config["dpi"], bool) and isinstance(config["dpi"], int)
             and config["dpi"] > 0, "dpi must be a positive integer")
    size = np.asarray(config["figure_size"], dtype=float)
    _require(size.shape == (2,) and np.isfinite(size).all() and np.all(size > 0), "figure_size must contain two positive values")
    config["figure_size"] = size.tolist()
    _require(isinstance(config["directory_name"], str) and config["directory_name"]
             and Path(config["directory_name"]).name == config["directory_name"]
             and config["directory_name"] not in (".", ".."), "directory_name must be one safe directory name")
    for name in ("shade_min", "shade_max"):
        config[name] = _real(config[name], name)
    _require(0 <= config["shade_min"] < config["shade_max"] <= 1, "Color shade bounds must be ordered within [0,1]")
    if "torch_num_threads" in config:
        _require(not isinstance(config["torch_num_threads"], bool) and isinstance(config["torch_num_threads"], int)
                 and config["torch_num_threads"] > 0, "torch_num_threads must be a positive integer")
    return config


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _policy_digest(payload):
    digest = hashlib.sha256()
    for network in ("actor", "critic"):
        state = payload.get(network)
        _require(isinstance(state, dict) and state, "Checkpoint is missing " + network)
        for name, value in sorted(state.items()):
            _require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()),
                     "Non-finite or invalid checkpoint parameter: " + network + "." + name)
            tensor = value.detach().cpu().contiguous()
            digest.update((network + "." + name + str(tensor.dtype) + str(tuple(tensor.shape))).encode("utf-8"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _priority(path):
    name = path.name
    if name.startswith("evaluated_"):
        rank = 0
    elif name in ("final.pt", "last.pt"):
        rank = 1
    elif name in ("initial.pt", "best.pt", "best_eval.pt"):
        rank = 2
    else:
        rank = 3
    return rank, path.as_posix()


def discover_checkpoints(run_dir):
    """Find this output's saved policies without descending into other runs.

    Later-stage initial checkpoints are transfer diagnostics, not newly trained
    policies at their global counters; they are excluded to avoid two different
    strategies appearing at the same training step after a best-eval handoff.
    """
    run = _resolve(run_dir)
    _require(run.is_dir(), "Training output directory does not exist: " + str(run))
    directories = [run / "checkpoints"]
    directories += [path / "checkpoints" for path in sorted(run.glob("stage_*")) if path.is_dir()]
    paths = sorted({path.resolve() for directory in directories if directory.is_dir()
                    for path in directory.glob("*.pt") if path.is_file()})
    _require(bool(paths), "No checkpoint files found directly in this training output")
    by_step, signature, excluded = {}, None, []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(payload.get("format_version") == 1, "Unsupported checkpoint format: " + str(path))
        state = payload.get("training_state", {})
        if path.name == "resume_initial.pt" or (path.name == "initial.pt" and state.get("stage_index", 1) > 1):
            excluded.append(dict(checkpoint=str(path), reason="Stage/continuation handoff diagnostic; not a newly trained policy at this actual-work counter"))
            continue
        step = state.get("env_steps")
        _require(not isinstance(step, bool) and isinstance(step, (int, np.integer)) and step >= 0,
                 "Checkpoint lacks an actual nonnegative env_steps counter: " + str(path))
        config = payload.get("config", {})
        environment = payload.get("metadata", {}).get("environment_config")
        _require(isinstance(environment, dict), "Checkpoint has no environment snapshot: " + str(path))
        _require(all(isinstance(config.get(name), dict) for name in ("network", "ppo", "normalization")),
                 "Checkpoint lacks its learning configuration: " + str(path))
        actual_signature = _canonical(dict(environment=environment, dimensions=payload.get("dimensions"),
                                           **{name: config[name] for name in ("network", "ppo", "normalization")}))
        if signature is None:
            signature = actual_signature
        _require(actual_signature == signature,
                 "Mixed environment/network/PPO/normalization settings in checkpoint inventory: " + str(path))
        policy_digest = _policy_digest(payload)
        record = dict(step=int(step), checkpoint=path, checkpoint_sha256=_sha(path), aliases=[path],
                      alias_sha256={str(path): _sha(path)}, environment_config=deepcopy(environment),
                      config=deepcopy(config), signature=actual_signature, policy_sha256=policy_digest)
        if int(step) in by_step:
            previous = by_step[int(step)]
            _require(previous["policy_sha256"] == policy_digest,
                     "Ambiguous different policies at the same env_steps={}: {} and {}".format(step, previous["checkpoint"], path))
            aliases = previous["aliases"] + [path]
            hashes = dict(previous["alias_sha256"], **record["alias_sha256"])
            if _priority(path) < _priority(previous["checkpoint"]):
                previous = record
            previous["aliases"], previous["alias_sha256"] = aliases, hashes
            by_step[int(step)] = previous
        else:
            by_step[int(step)] = record
    _require(bool(by_step), "No trained or original-initial checkpoints are available")
    records = [by_step[key] for key in sorted(by_step)]
    records[0]["excluded_checkpoints"] = excluded
    return records


def select_checkpoints(checkpoints, target_steps, extra_steps=None):
    """Choose nearest real checkpoints, plus endpoints and saved best policies."""
    targets = _steps(target_steps, "target_steps")
    extras = _steps([] if extra_steps is None else extra_steps, "extra_steps")
    _require(bool(checkpoints), "At least one real checkpoint is required")
    ordered = sorted(checkpoints, key=lambda item: item["step"])
    available = {record["step"]: record for record in ordered}
    chosen = {ordered[0]["step"], ordered[-1]["step"]}
    mapping = []
    for target in targets:
        actual = min(available, key=lambda value: (abs(value - target), value))
        chosen.add(actual)
        mapping.append(dict(requested_step=target, actual_step=actual,
                            checkpoint=str(available[actual]["checkpoint"]), selection="nearest_available"))
    for target in extras:
        _require(target in available, "Explicit extra checkpoint step is unavailable: " + str(target))
        chosen.add(target)
        mapping.append(dict(requested_step=target, actual_step=target,
                            checkpoint=str(available[target]["checkpoint"]), selection="explicit_extra"))
    for record in ordered:
        if any(path.name in ("best.pt", "best_eval.pt") for path in record["aliases"]):
            chosen.add(record["step"])
    return [available[key] for key in sorted(chosen)], mapping


def _rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _protected_sources(run, inventory):
    sources = set()
    for item in inventory:
        sources.update(item["aliases"])
    for name in ("metadata.json", "training_config.yaml", "environment_config.yaml", "summary.json",
                 "curriculum_state.json", "evaluation_history.csv", "training_metrics.csv", "updates.csv"):
        if (run / name).is_file():
            sources.add(run / name)
    sources.update(run.glob("stage_*/**/*.csv"))
    return {str(path.resolve()): _sha(path) for path in sorted(sources)}


def generate_trajectory_evolution(run_dir, *, config_path=None, output_dir=None, extra_steps=None):
    """Run one full deterministic episode per selected policy and save evidence."""
    config = load_evolution_config(config_path)
    run = _resolve(run_dir)
    output = _resolve(output_dir) if output_dir is not None else run / config["directory_name"]
    if not config["enabled"]:
        return dict(status="disabled", output_dir=str(output), plots={})
    _require(not output.exists(), "Refusing to overwrite existing trajectory evolution output: " + str(output))
    # Discovery performs no construction or global seeding. Capture state before
    # even loading checkpoints so plotting/validation failures are isolated too.
    original_rng = _rng_snapshot()
    original_threads = torch.get_num_threads()
    original_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        # Small state dictionaries benefit from the configured CPU thread count
        # during inventory validation too, not only during actor inference.
        torch.set_num_threads(config.get("torch_num_threads", original_threads))
        inventory = discover_checkpoints(run)
        extras = config["extra_checkpoint_steps"] + _steps([] if extra_steps is None else extra_steps, "extra_steps")
        selected, mapping = select_checkpoints(inventory, config["target_steps"], extras)
        hashes_before = _protected_sources(run, inventory)
        environment = deepcopy(inventory[0]["environment_config"])
        _require("mission" in environment, "Fixed azimuth evolution requires a point-defense environment")
        original_initialization = deepcopy(environment["initialization"])
        environment["initialization"]["attacker_position"] = arrival_position(environment, config["angle_degrees"]).tolist()
        output.mkdir(parents=True)
        trajectories = output / "trajectories"
        trajectories.mkdir()
        scene_path = output / "fixed_environment_config.yaml"
        scene_path.write_text(yaml.safe_dump(environment, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (output / "evolution_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        episodes, rows, initial_states = [], [], None
        for item in selected:
            thread_count = config.get("torch_num_threads", item["config"]["torch_num_threads"])
            set_seed(config["seed"], thread_count)
            _require(_sha(item["checkpoint"]) == item["checkpoint_sha256"],
                     "Checkpoint file changed after inventory: " + str(item["checkpoint"]))
            agent, payload = MAPPOAgent.from_checkpoint(item["checkpoint"], device=config["device"])
            _require(_policy_digest(payload) == item["policy_sha256"], "Checkpoint changed after inventory")
            agent.actor.eval()
            agent.critic.eval()
            env = PursuitEnv(scene_path)
            with torch.inference_mode():
                metrics, trajectory = run_episode(env, agent, config["seed"])
            for name, array in trajectory.items():
                if np.issubdtype(array.dtype, np.number):
                    _require(np.isfinite(array).all(), "Non-finite saved trajectory: " + name)
            _require(len(trajectory["time"]) == metrics["episode_length"] + 1, "Incomplete episode trajectory")
            if initial_states is None:
                initial_states = trajectory["states"][0].copy()
            _require(np.array_equal(initial_states, trajectory["states"][0]), "Test initial states differ between checkpoints")
            source = trajectories / "step_{:06d}.npz".format(item["step"])
            save_trajectory(source, trajectory, metrics, config["seed"])
            result = "Capture" if metrics["captured"] else "Timeout" if metrics["truncated"] else "Fail"
            rows.append(dict(step=item["step"], result=result, captured=bool(metrics["captured"]),
                             episode_length=metrics["episode_length"], event_time=float(trajectory["time"][-1]),
                             min_pursuer_attacker_distance=metrics["min_distance"],
                             final_attacker_base_distance=metrics["attacker_base_terminal_distance"],
                             episode_reward=metrics["episode_reward"], checkpoint=str(item["checkpoint"]),
                             checkpoint_sha256=item["checkpoint_sha256"], trajectory=str(source)))
            episodes.append(dict(step=item["step"], checkpoint=str(item["checkpoint"]),
                                 checkpoint_sha256=item["checkpoint_sha256"], metrics=metrics, trajectory=trajectory))
            print("Trajectory evolution step={} result={} length={} min_distance={:.3f}m".format(
                item["step"], result, metrics["episode_length"], metrics["min_distance"]), flush=True)
            del agent, payload, env
        csv_path = output / "trajectory_evolution_metrics.csv"
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        from utils.trajectory_evolution_plot import plot_evolution
        plot_config = dict(config, run_label=run.name)
        plots = {name: str(path) for name, path in plot_evolution(episodes, output, plot_config).items()}
        hashes_after = {name: _sha(Path(name)) for name in hashes_before}
        _require(hashes_after == hashes_before, "Original model/configuration/metrics changed during evolution analysis")
        zero_available = any(item["step"] == 0 for item in inventory)
        manifest = dict(
            status="completed", created_at_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(run), output_dir=str(output),
            deterministic=True, actor_mode="eval", critic_mode="eval", inference_mode=True, training_performed=False,
            requested_steps=config["target_steps"], extra_checkpoint_steps=extras,
            selected_steps=[item["step"] for item in selected], sampling_mapping=mapping,
            excluded_checkpoints=inventory[0].get("excluded_checkpoints", []),
            initial_checkpoint_available=zero_available,
            zero_step_note=("A real saved 0-step checkpoint was evaluated." if zero_available else
                            "No 0-step checkpoint is available; the earliest real checkpoint is {} steps. No initial model was reconstructed.".format(inventory[0]["step"])),
            scene=dict(angle_degrees=config["angle_degrees"], seed=config["seed"], initial_state=initial_states.tolist(),
                       original_initialization=original_initialization, fixed_environment_config=str(scene_path),
                       fixed_environment_sha256=_sha(scene_path), source="common checkpoint metadata.environment_config snapshot"),
            source_checkpoints=[dict(step=item["step"], checkpoint=str(item["checkpoint"]),
                                     checkpoint_sha256=item["checkpoint_sha256"], policy_sha256=item["policy_sha256"],
                                     aliases=[str(path) for path in item["aliases"]]) for item in selected],
            source_files_unchanged=True, protected_source_sha256_before=hashes_before,
            protected_source_sha256_after=hashes_after, metrics_csv=str(csv_path), plots=plots,
            distance_semantics="3D Euclidean distances; XY plot is a projection, not the capture test",
            interpretation="One fixed deterministic episode per saved policy illustrates training history; it is not a capture-rate or generalization estimate.",
        )
        write_json(output / "trajectory_evolution_manifest.json", manifest)
        return manifest
    finally:
        torch.set_num_threads(original_threads)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = original_flags
        restore_rng_state(original_rng)
