"""Frozen-policy original/CCW90 episodes with physical control diagnostics."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from algorithms.mappo import MAPPOAgent
from envs import PursuitEnv
from envs.dynamics import scale_action
from scripts.evaluate_attacker_position_generalization import _frozen_hashes
from utils.evaluation import checkpoint_provenance, run_episode, save_trajectory, write_json
from utils.experiment import ROOT, set_seed, write_metadata


DEFAULT_CONFIG = ROOT / "configs/eval_direction_ccw90.yaml"


def _path(value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ControlRecordingEnv:
    """Observe controls around the unmodified environment's reset/step calls."""

    def __init__(self, environment):
        self.environment = environment

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def reset(self, seed=None):
        observations, info = self.environment.reset(seed=seed)
        self.current_observations = observations.copy()
        self.observations_history = []
        self.command_history = []
        self.applied_history = []
        self.realized_history = []
        self.aircraft_acceleration_history = [self.environment.accelerations.copy()]
        return observations, info

    def step(self, actions):
        env = self.environment
        self.observations_history.append(self.current_observations.copy())
        command = scale_action(actions, env.pursuer_limits)
        previous_velocity = env.states[:env.num_pursuers, 3:].copy()
        result = env.step(actions)
        self.current_observations = result[0].copy()
        self.command_history.append(command)
        self.applied_history.append(env.accelerations[:env.num_pursuers].copy())
        self.realized_history.append((env.states[:env.num_pursuers, 3:] - previous_velocity) / env.dt)
        self.aircraft_acceleration_history.append(env.accelerations.copy())
        return result


def record_episode(environment, agent, seed):
    recorder = ControlRecordingEnv(environment)
    metrics, trajectory = run_episode(recorder, agent, seed)
    trajectory.update(
        observations=np.asarray(recorder.observations_history),
        command_accelerations=np.asarray(recorder.command_history),
        applied_accelerations=np.asarray(recorder.applied_history),
        realized_accelerations=np.asarray(recorder.realized_history),
        aircraft_accelerations=np.asarray(recorder.aircraft_acceleration_history),
    )
    for value in trajectory.values():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RuntimeError("Non-finite recorded trajectory/control")
    return metrics, trajectory


def control_rows(name, trajectory):
    for step in range(len(trajectory["actions"])):
        for agent in range(3):
            row = dict(scenario=name, step=step + 1, time_before=float(trajectory["time"][step]),
                       time_after=float(trajectory["time"][step + 1]), agent=agent)
            for prefix, array in (("action", trajectory["actions"]),
                                  ("command", trajectory["command_accelerations"]),
                                  ("applied", trajectory["applied_accelerations"]),
                                  ("realized", trajectory["realized_accelerations"])):
                row.update({prefix + "_" + axis: float(array[step, agent, index])
                            for index, axis in enumerate("xyz")})
            for index, axis in enumerate("xyz"):
                row["position_" + axis] = float(trajectory["positions"][step + 1, agent, index])
                row["velocity_" + axis] = float(trajectory["states"][step + 1, agent, index + 3])
                row["observed_attacker_relative_" + axis] = float(trajectory["observations"][step, agent, index + 3])
            yield row


def _csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config_path=DEFAULT_CONFIG, output_dir=None):
    config_path = Path(config_path).resolve()
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = _path(protocol["checkpoint"])
    digest = _sha(checkpoint)
    if digest != protocol["expected_checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA differs from the confirmed 92160-step model")
    position = np.asarray(protocol["attacker_position"], dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("attacker_position must have three finite coordinates")
    seed = protocol["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = _path(protocol["output_dir"] if output_dir is None else output_dir)
    output.relative_to(ROOT)
    artifacts = ("baseline", "ccw90", "metadata.json", "summary.json", "controls.csv", "evaluation_protocol.yaml")
    if output.exists() and (not output.is_dir() or any((output / n).exists() for n in artifacts)):
        raise ValueError("Refusing to overwrite existing evaluation artifacts: " + str(output))
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    agent, payload = MAPPOAgent.from_checkpoint(checkpoint, device="cpu")
    if payload["training_state"]["env_steps"] != protocol["expected_env_steps"]:
        raise ValueError("Checkpoint step count differs from confirmed model")
    baseline = deepcopy(payload["metadata"]["environment_config"])
    reference_config = _path(protocol["reference_environment_config"])
    if baseline != yaml.safe_load(reference_config.read_text(encoding="utf-8")):
        raise ValueError("Reference YAML differs from checkpoint's environment snapshot")
    if baseline["attacker"]["policy"]["mode"] != "direct":
        raise ValueError("Expected the unchanged Direct Attacker")
    if np.any(position < baseline["environment"]["bounds"]["low"]) or np.any(position > baseline["environment"]["bounds"]["high"]):
        raise ValueError("Requested initial position exceeds workspace")
    set_seed(seed, payload["config"]["torch_num_threads"])
    for network in (agent.actor, agent.critic):
        network.eval()
        network.requires_grad_(False)
    before = _frozen_hashes(agent)
    reference = _path(protocol["reference_trajectory"])
    reference_sha = _sha(reference)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summaries, all_rows, reference_arrays = {}, [], []
    with torch.inference_mode():
        for name in ("baseline", "ccw90"):
            scenario = deepcopy(baseline)
            if name == "ccw90":
                scenario["initialization"]["attacker_position"] = position.tolist()
            folder = output / name
            folder.mkdir()
            scenario_path = folder / "environment_config.yaml"
            scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True), encoding="utf-8")
            env = PursuitEnv(scenario_path)
            metrics, trajectory = record_episode(env, agent, seed)
            expected = np.vstack((baseline["initialization"]["pursuer_positions"], scenario["initialization"]["attacker_position"]))
            np.testing.assert_array_equal(trajectory["positions"][0], expected)
            np.testing.assert_array_equal(trajectory["states"][0, :, 3:], np.zeros((4, 3)))
            np.testing.assert_array_equal(trajectory["aircraft_accelerations"][0], np.zeros((4, 3)))
            if name == "baseline":
                with np.load(reference, allow_pickle=False) as old:
                    reference_arrays = sorted(set(old.files) - {"metadata_json"})
                    for key in reference_arrays:
                        np.testing.assert_array_equal(trajectory[key], old[key])
            save_trajectory(folder / "trajectory.npz", trajectory, metrics, seed)
            write_json(folder / "metrics.json", metrics)
            rows = list(control_rows(name, trajectory))
            _csv(folder / "controls.csv", rows)
            all_rows.extend(rows)
            summaries[name] = metrics
            print("{}: outcome={}, steps={}, min_distance={:.6f}m, capture_time={}".format(
                name, metrics["outcome"], metrics["episode_length"], metrics["min_distance"], metrics["capture_time"]), flush=True)
    after = _frozen_hashes(agent)
    if before != after or _sha(checkpoint) != digest or _sha(reference) != reference_sha:
        raise RuntimeError("Frozen model/optimizer/RNG or original artifacts changed")
    _csv(output / "controls.csv", all_rows)
    freeze = dict(before=before, after=after, all_unchanged=True,
                  all_parameters_require_grad_false=all(not p.requires_grad for n in (agent.actor, agent.critic) for p in n.parameters()),
                  actor_eval_mode=not agent.actor.training, critic_eval_mode=not agent.critic.training,
                  original_reference_arrays=reference_arrays, original_reference_bit_identical=True)
    metadata = write_metadata(
        output / "metadata.json", payload["config"], baseline,
        config_path=config_path, env_config_path=reference_config, output_dir=output,
        kind="direction_generalization_evaluation", extra={
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
            "checkpoint_training_state": checkpoint_provenance(payload),
            "evaluation_protocol": protocol, "deterministic": True, "no_training": True,
            "only_environment_change": "initialization.attacker_position",
            "actor_input_or_action_rotation_compensation": False,
            "frozen_state_verification": freeze,
            "source_sha256": {Path(__file__).resolve().relative_to(ROOT).as_posix(): _sha(__file__)},
            "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - clock,
            "control_definitions": {"action": "Actor tanh output, dimensionless [-1,1]",
                                    "command": "scale_action output before jerk, m/s^2",
                                    "applied": "jerk-limited command stored by env, m/s^2",
                                    "realized": "post-speed-limit velocity difference / dt, m/s^2"},
        })
    summary = dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=digest,
                   checkpoint_training_steps=payload["training_state"]["env_steps"], scenarios=summaries,
                   formal_episodes=2, rotated_episodes=1, no_training=True,
                   completed_at_utc=metadata["completed_at_utc"])
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluate(args.config, output_dir=args.output_dir)
