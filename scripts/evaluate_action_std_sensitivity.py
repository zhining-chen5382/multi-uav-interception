"""Frozen-policy pre-tanh action-std sensitivity; no training or config override.

Reuse the existing environment, episode metrics and trajectory recording. A
read-only actor adapter owns the sampling law so no network/log_std/optimizer
or core MAPPO code needs to change. Each multiplier uses the same episode seeds
and the same standard-Normal noise prefixes (common random numbers).
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import yaml

from algorithms.mappo import MAPPOAgent
from envs import PursuitEnv
from scripts.evaluate_attacker_position_generalization import (
    _frozen_hashes, _sha, _state_sha,
)
from utils.evaluation import (
    checkpoint_provenance, run_episode, save_trajectory, summarize_episodes, write_json,
)
from utils.experiment import ROOT, write_metadata


DEFAULT_CONFIG = ROOT / "configs/eval_action_std_sensitivity.yaml"


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _multiplier(value):
    if isinstance(value, (bool, str, complex)):
        raise ValueError("std multiplier must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("std multiplier must be a finite nonnegative number")
    if not np.isfinite(result) or result < 0:
        raise ValueError("std multiplier must be a finite nonnegative number")
    return result


def validate_protocol(config):
    if not isinstance(config, dict):
        raise ValueError("Evaluation protocol must be a mapping")
    _integer(config["episodes_per_group"], "episodes_per_group", 1)
    base_seed = _integer(config["base_seed"], "base_seed")
    if base_seed + config["episodes_per_group"] > 2 ** 63:
        raise ValueError("Episode seeds exceed torch.Generator range")
    groups = config["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list")
    names, multipliers = set(), set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Each group must be a mapping")
        name = group.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("Group names must be safe directory identifiers")
        multiplier = _multiplier(group["multiplier"])
        if name in names or multiplier in multipliers:
            raise ValueError("Group names and multipliers must be unique")
        names.add(name)
        multipliers.add(multiplier)
    if config.get("representative_selection", "lower_median_min_distance") != "lower_median_min_distance":
        raise ValueError("Unsupported representative selection")


class FrozenStdSampler:
    """Run the unchanged actor with a local, seeded pre-tanh noise multiplier.

    The shared run_episode evaluator passes deterministic=True to act(). This
    adapter deliberately owns its explicitly configured law instead: only its
    multiplier==0 is deterministic. No global RNG is consumed and no parameter,
    optimizer, gradient or network mode is changed by this adapter.
    """

    def __init__(self, agent, multiplier, seed):
        self.agent = agent
        self.multiplier = _multiplier(multiplier)
        _integer(seed, "seed")
        if str(agent.device) != "cpu":
            raise ValueError("This paired-noise evaluation requires CPU inference")
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.history = {name: [] for name in ("raw_actions", "means", "noise", "base_stds")}

    @torch.no_grad()
    def act(self, observations, global_state=None, deterministic=True):
        observations = self.agent._tensor(observations)
        if observations.ndim != 2 or observations.shape[-1] != self.agent.observation_dim:
            raise ValueError("observations must have shape (num_agents, observation_dim)")
        distribution = self.agent.actor.distribution(observations)
        mean, std = distribution.mean, distribution.scale
        noise = (torch.zeros_like(mean) if self.multiplier == 0 else torch.randn(
            mean.shape, generator=self.generator, dtype=mean.dtype, device=mean.device))
        raw = mean if self.multiplier == 0 else mean + self.multiplier * std * noise
        action = raw.tanh()
        if not all(torch.isfinite(value).all() for value in (mean, std, noise, raw, action)):
            raise FloatingPointError("Non-finite frozen actor sampling output")
        for name, value in (("raw_actions", raw), ("means", mean),
                            ("noise", noise), ("base_stds", std)):
            self.history[name].append(value.cpu().numpy().copy())
        return action.cpu().numpy(), None, None, None

    def sampling_report(self):
        actor = self.agent.actor
        with torch.no_grad():
            effective = actor.log_std.clamp(actor.min_log_std, actor.max_log_std)
            base = effective.exp()
            scaled = base * self.multiplier
        return {"log_std": actor.log_std.detach().cpu().tolist(),
                "effective_log_std": effective.cpu().tolist(), "base_std": base.cpu().tolist(),
                "multiplier": self.multiplier, "scaled_std": scaled.cpu().tolist()}


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Distribution requires nonempty finite one-dimensional data")
    quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return dict(mean=float(values.mean()), std=float(values.std(ddof=0)),
                min=float(values.min()), max=float(values.max()),
                **dict(zip(("p10", "p25", "median", "p75", "p90"), map(float, quantiles))))


def wilson_interval(successes, count):
    z = 1.959963984540054
    fraction = successes / count
    center = (fraction + z * z / (2 * count)) / (1 + z * z / count)
    half = z * np.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count ** 2)) / (1 + z * z / count)
    return [float(max(0, center - half)), float(min(1, center + half))]


def representatives(rows):
    def choose(subset, label):
        ordered = sorted(subset, key=lambda row: (row["min_distance"], row["episode_index"]))
        row = ordered[(len(ordered) - 1) // 2]
        return {"episode_index": row["episode_index"], "seed": row["seed"],
                "trajectory": row["trajectory"], "selection": label,
                "metrics": {key: value for key, value in row.items()
                            if key not in ("group", "multiplier", "episode_index", "seed", "trajectory")}}
    result = {"typical": choose(rows, "lower median minimum distance over ALL episodes")}
    for label, captured in (("success", True), ("failure", False)):
        subset = [row for row in rows if row["captured"] == captured]
        if subset:
            result[label] = choose(subset, "lower median minimum distance among " + label + " episodes")
    return result


def evaluate(config_path=DEFAULT_CONFIG, output_dir=None):
    """Run all frozen groups, retaining every actual trajectory and random seed."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    checkpoint = _path(config["checkpoint"])
    source_sha = _sha(checkpoint)
    if source_sha != config["expected_checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from the confirmed v08 model")
    output = _path(config["output_dir"] if output_dir is None else output_dir)
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment: " + str(output))
    started, start_clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    reference_config = _path(config["reference_environment_config"])
    reference_trajectory = _path(config["reference_trajectory"])
    protected = {str(path): _sha(path) for path in (checkpoint, config_path, reference_config, reference_trajectory)}
    old_threads = torch.get_num_threads()
    global_rng_before = _state_sha(torch.get_rng_state())
    rows, summaries, selected = [], {}, {}
    try:
        # Loading allocates/initializes networks. Restore the caller's CPU RNG
        # afterwards; the experiment itself uses only per-episode generators.
        with torch.random.fork_rng(devices=[]):
            agent, payload = MAPPOAgent.from_checkpoint(checkpoint, device="cpu")
            if payload["training_state"]["env_steps"] != config["expected_env_steps"]:
                raise ValueError("Wrong checkpoint training step")
            base = deepcopy(payload["metadata"]["environment_config"])
            if base != yaml.safe_load(reference_config.read_text(encoding="utf-8")):
                raise ValueError("Reference environment differs from checkpoint snapshot")
            if base.get("attacker", {}).get("policy", {}).get("mode") != "direct":
                raise ValueError("Expected unchanged Direct Attacker")
            initial = base["initialization"]["attacker_position"]
            if initial[0] <= 0 or initial[1] != 0:
                raise ValueError("Expected the original fixed 0 degree scenario")
            torch.set_num_threads(payload["config"]["torch_num_threads"])
            for network in (agent.actor, agent.critic):
                network.eval()
                network.requires_grad_(False)
            before = _frozen_hashes(agent)
            output.mkdir(parents=True, exist_ok=False)
            for filename, mapping in (("evaluation_protocol.yaml", config),
                                      ("environment_config.yaml", base),
                                      ("source_training_config.yaml", payload["config"])):
                (output / filename).write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
            env = PursuitEnv(output / "environment_config.yaml")
            first_initial_state, fixed_reference_names = None, []
            fixed_states_hash = None
            action_std = FrozenStdSampler(agent, 1, config["base_seed"]).sampling_report()
            for group in config["groups"]:
                name, multiplier = group["name"], float(group["multiplier"])
                folder = output / "groups" / name
                folder.mkdir(parents=True)
                group_rows = []
                for index in range(config["episodes_per_group"]):
                    seed = config["base_seed"] + index
                    sampler = FrozenStdSampler(agent, multiplier, seed)
                    metrics, trajectory = run_episode(env, sampler, seed)
                    trajectory.update({"latent_" + key: np.asarray(values)
                                       for key, values in sampler.history.items()})
                    if first_initial_state is None:
                        first_initial_state = trajectory["states"][0].copy()
                    if not np.array_equal(trajectory["states"][0], first_initial_state):
                        raise AssertionError("Initial state changed across groups or episodes")
                    for key, value in trajectory.items():
                        if np.issubdtype(np.asarray(value).dtype, np.number) and not np.isfinite(value).all():
                            raise FloatingPointError("Non-finite trajectory array " + key)
                    if multiplier == 0:
                        signature = _state_sha({key: trajectory[key] for key in ("states", "actions", "rewards")})
                        if fixed_states_hash is None:
                            fixed_states_hash = signature
                            with np.load(reference_trajectory, allow_pickle=False) as reference:
                                fixed_reference_names = sorted(set(reference.files).intersection(trajectory) - {"metadata_json"})
                                for key in fixed_reference_names:
                                    if not np.array_equal(trajectory[key], reference[key]):
                                        raise AssertionError("Deterministic source reproduction differs: " + key)
                        elif signature != fixed_states_hash:
                            raise AssertionError("Zero-std episodes should repeat exactly")
                    path = folder / ("episode_{:03d}.npz".format(index))
                    save_trajectory(path, trajectory, metrics, seed)
                    row = dict(group=name, multiplier=multiplier, episode_index=index, seed=seed,
                               trajectory=path.relative_to(output).as_posix(), **metrics)
                    rows.append(row)
                    group_rows.append(row)
                    if (index + 1) % 25 == 0:
                        print("{} std={}x: {}/{} episodes, captures={}".format(
                            name, multiplier, index + 1, config["episodes_per_group"],
                            sum(row["captured"] for row in group_rows)), flush=True)
                _csv(folder / "episode_metrics.csv", group_rows)
                summary = dict(summarize_episodes(group_rows), multiplier=multiplier,
                               min_distance_distribution=distribution_summary([r["min_distance"] for r in group_rows]),
                               episode_reward_distribution=distribution_summary([r["episode_reward"] for r in group_rows]),
                               capture_rate_wilson_95=wilson_interval(sum(r["captured"] for r in group_rows), len(group_rows)),
                               action_std=FrozenStdSampler(agent, multiplier, config["base_seed"]).sampling_report(),
                               seed_range=[config["base_seed"], config["base_seed"] + len(group_rows) - 1],
                               deterministic=multiplier == 0)
                summaries[name] = summary
                selected[name] = representatives(group_rows)
                write_json(folder / "summary.json", summary)
            after = _frozen_hashes(agent)
            if before != after:
                raise RuntimeError("Frozen networks/optimizers or global Torch RNG changed")
            for path, digest in protected.items():
                if _sha(Path(path)) != digest:
                    raise RuntimeError("Source artifact changed: " + path)
            verification = dict(all_checks_passed=True, before=before, after=after,
                                all_parameters_require_grad_false=all(not p.requires_grad for n in (agent.actor, agent.critic) for p in n.parameters()),
                                all_parameter_gradients_none=all(p.grad is None for n in (agent.actor, agent.critic) for p in n.parameters()),
                                all_initial_states_identical=True, source_artifacts_sha256=protected,
                                fixed_first_episode_reference_arrays_identical=fixed_reference_names,
                                zero_multiplier_all_episodes_bit_identical=True,
                                no_training=True, no_optimizer_steps=True, log_std_unchanged=True,
                                seed_pairing="Same episode seed across multipliers; local CPU torch.Generator, identical standard-Normal prefixes")
            _csv(output / "episode_metrics.csv", rows)
            write_json(output / "representatives.json", selected)
            result = dict(episodes_total=len(rows), checkpoint=str(checkpoint), checkpoint_sha256=source_sha,
                          checkpoint_training_steps=payload["training_state"]["env_steps"],
                          groups=summaries, action_std=action_std, angle_degrees=0.0,
                          sampling_formula="action = tanh(mean(obs) + multiplier * exp(clamp(log_std)) * epsilon); epsilon ~ N(0,I)",
                          std_units="Pre-tanh latent Normal standard deviations; not final action or physical acceleration standard deviations",
                          deterministic_note="0x repeats the exact same scenario and trajectory 100 times; not independent generalization evidence",
                          frozen_state_verification=verification,
                          started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                          wall_seconds=time.perf_counter() - start_clock)
            training_config = deepcopy(payload["config"])
            training_config["seed"] = config["base_seed"]
            write_metadata(output / "metadata.json", training_config, base, config_path=config_path,
                           env_config_path=reference_config, output_dir=output, kind="frozen_action_std_evaluation",
                           extra={"evaluation_protocol": config, "no_training": True,
                                  "checkpoint": str(checkpoint), "checkpoint_sha256": source_sha,
                                  "checkpoint_training_state": checkpoint_provenance(payload),
                                  "started_at_utc": started, "completed_at_utc": result["completed_at_utc"],
                                  "frozen_state_verification": verification})
    finally:
        torch.set_num_threads(old_threads)
    result["caller_global_torch_rng_restored"] = global_rng_before == _state_sha(torch.get_rng_state())
    if not result["caller_global_torch_rng_restored"]:
        raise RuntimeError("Caller global Torch RNG was not restored")
    write_json(output / "summary.json", result)
    print(json.dumps({"episodes_total": len(rows), "capture_rates": {
        key: value["capture_rate"] for key, value in summaries.items()}, "output_dir": str(output),
        "frozen_verified": True}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluate(args.config, args.output_dir)


if __name__ == "__main__":
    main()
