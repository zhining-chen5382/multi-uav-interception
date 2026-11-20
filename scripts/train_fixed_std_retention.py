"""Fixed-zero policy retention with fixed pre-tanh action standard deviation.

All groups inherit the same source networks, Adam moments/counters and global
RNG. Their only differing treatment is the initial std multiplier; log_std is
frozen in every group. The existing rollout/PPO/exact-evaluation loop is reused.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts import train_policy_retention as retention
from utils.curriculum_warm_start import load_warm_start_source
from utils.evaluation import write_json
from utils.experiment import ROOT, load_training_config, set_seed, write_metadata


DEFAULT_CONFIG = ROOT / "configs/mappo_fixed_std_retention.yaml"


def validate_config(config, environment_config):
    """Check all treatments against the unchanged original physical/PPO setup."""
    retention._require(isinstance(config.get("retention"), dict), "retention must be a mapping")
    for key in ("curriculum", "warm_start", "continuation", "initial_checkpoint", "resume_checkpoint",
                "scheduler", "learning_rate_scheduler"):
        retention._require(not config.get(key), "Fixed-std retention cannot enable " + key)
    settings = config["retention"]
    groups = settings.get("groups")
    retention._require(isinstance(groups, list) and len(groups) == 3,
                       "Fixed-std retention requires three A/B/C groups")
    multipliers = []
    for name, group in zip(("A", "B", "C"), groups):
        retention._require(isinstance(group, dict)
                           and set(group) == {"name", "reset_adam", "frozen_std_multiplier"}
                           and group["name"] == name and group["reset_adam"] is False,
                           "A/B/C must retain Adam and specify only their frozen std multiplier")
        multiplier = group["frozen_std_multiplier"]
        retention._require(not isinstance(multiplier, bool) and isinstance(multiplier, (int, float))
                           and np.isfinite(multiplier) and multiplier > 0,
                           "Frozen std multipliers must be finite positive numbers")
        multipliers.append(float(multiplier))
    retention._require(multipliers[0] == 1.0 and multipliers[0] > multipliers[1] > multipliers[2],
                       "A must fix the original std; B/C must use strictly decreasing lower std")
    retention._count(settings.get("evaluation_interval_steps"), "retention.evaluation_interval_steps")
    seed = settings.get("evaluation_seed")
    retention._require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
                       "retention.evaluation_seed must be a nonnegative integer")
    retention._count(config["training"].get("total_env_steps"), "training.total_env_steps")
    for key in ("rollout_steps", "log_interval_updates", "checkpoint_interval_updates", "capture_rate_window"):
        retention._count(config["training"].get(key), "training." + key)
    retention._require(environment_config.get("attacker", {}).get("policy", {}).get("mode") == "direct",
                       "Fixed-std retention requires the unchanged fixed Direct Attacker task")
    source_config = deepcopy(config)
    source_config["warm_start"] = {name:settings[name]
                                   for name in ("checkpoint", "expected_sha256") if name in settings}
    source = load_warm_start_source(source_config, environment_config)
    source_logs = source["payload"]["actor"]["log_std"]
    for multiplier in multipliers:
        shifted = source_logs + math.log(multiplier)
        retention._require(bool(torch.isfinite(shifted).all())
                           and bool((shifted >= config["network"]["min_log_std"]).all())
                           and bool((shifted <= config["network"]["max_log_std"]).all()),
                           "Requested std multiplier would be changed by the existing log_std clamp")
    return source


def run_experiment(config_path=DEFAULT_CONFIG, *, output_dir=None, seed=None):
    config_path = retention._resolve(config_path)
    config, env_path = load_training_config(config_path)
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if seed is not None:
        config["seed"] = seed
    environment = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    source = validate_config(config, environment)
    output = retention._resolve(config["output_dir"])
    retention._require(not output.exists(), "Use a new fixed-std retention output: " + str(output))
    original_run = source["checkpoint"].parent.parent
    retention._require(output != original_run and original_run not in output.parents,
                       "Fixed-std retention output must be independent of the source run")
    config["retention_config_source"] = str(config_path)
    set_seed(config["seed"], config["torch_num_threads"])
    output.mkdir(parents=True)
    write_metadata(output / "metadata.json", config, environment,
        config_path=config_path, env_config_path=env_path, output_dir=output,
        kind="fixed_action_std_retention_comparison",
        extra=dict(source_checkpoint=str(source["checkpoint"]), source_checkpoint_sha256=source["sha256"],
            source_pretrained_env_steps=source["source_env_steps"],
            treatment="Only initial std multiplier varies; log_std frozen in ALL arms; source Actor mean/Critic and both complete Adam states retained.",
            sampling_log_probability_entropy_use_same_actor_distribution=True,
            fixed_std_control_rerun=True,
            historical_trainable_std_control_is_not_a_fair_fixed_std_control=True,
            historical_trainable_std_reference=config["retention"].get("historical_trainable_std_reference"),
            actor_learning_rate=config["ppo"]["actor_learning_rate"],
            critic_learning_rate=config["ppo"]["critic_learning_rate"],
            learning_rate_scheduler_enabled=False))
    results = [retention.run_group(config, env_path, source, group, output / group["name"])
               for group in config["retention"]["groups"]]
    equality = {}
    reference_path = output / "A/evaluations/at_000000/first_episode.npz"
    with np.load(reference_path, allow_pickle=False) as reference:
        for name in ("B", "C"):
            with np.load(output / name / "evaluations/at_000000/first_episode.npz", allow_pickle=False) as actual:
                equality[name] = {key:np.array_equal(reference[key], actual[key])
                                  for key in reference.files if key != "metadata_json"}
    retention._require(all(all(checks.values()) for checks in equality.values()),
                       "Initial deterministic trajectories changed across std treatments")
    retention._require(all(result["frozen_std_verification"]["all_checks_passed"] for result in results),
                       "Fixed-std training verification failed")
    summary = dict(status="completed", output_dir=str(output), groups=results,
        source_checkpoint_sha256=source["sha256"],
        source_unchanged=retention._sha(source["checkpoint"]) == source["sha256"],
        fixed_std_multipliers={group["name"]:group["frozen_std_multiplier"] for group in config["retention"]["groups"]},
        initial_deterministic_array_equality=equality,
        stochastic_training_rollouts_are_expected_to_differ=True,
        original_adam_states_retained=True, log_std_fixed_in_all_groups=True,
        source_log_std_adam_moments_and_counters_unchanged=True,
        actor_learning_rate=config["ppo"]["actor_learning_rate"],
        critic_learning_rate=config["ppo"]["critic_learning_rate"],
        fixed_std_control_rerun=True, historical_trainable_std_control_only_a_reference=True,
        curriculum_training_started=False, learning_rate_scheduler_enabled=False,
        source_pretrained_env_steps=source["source_env_steps"],
        total_added_work_steps=sum(result["env_steps"] for result in results))
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if args.check_config:
        config, env_path = load_training_config(args.config)
        if args.output_dir is not None:
            config["output_dir"] = str(args.output_dir)
        if args.seed is not None:
            config["seed"] = args.seed
        source = validate_config(config, yaml.safe_load(env_path.read_text(encoding="utf-8")))
        print(json.dumps(dict(output_dir=config["output_dir"], total_added_steps_per_group=config["training"]["total_env_steps"],
            source_checkpoint=str(source["checkpoint"]), source_sha256=source["sha256"],
            groups=config["retention"]["groups"], evaluation_interval_steps=config["retention"]["evaluation_interval_steps"],
            actor_learning_rate=config["ppo"]["actor_learning_rate"],
            critic_learning_rate=config["ppo"]["critic_learning_rate"],
            log_std_frozen_in_all_groups=True, fixed_std_control_rerun=True,
            formal_training_started=False, learning_rate_scheduler_enabled=False), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(run_experiment(args.config, output_dir=args.output_dir, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
