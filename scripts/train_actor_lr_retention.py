"""Fixed-zero Actor-LR ablation with source weights and both Adam states retained.

The shared retention runner owns sampling, PPO updates, exact evaluations and
saved trajectories. This entry point changes only the loaded Actor optimizer's
learning rate for each declared arm; it does not introduce a scheduler.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

from scripts import train_policy_retention as retention
from utils.curriculum_warm_start import load_warm_start_source
from utils.evaluation import write_json
from utils.experiment import ROOT, load_training_config, set_seed, write_metadata


DEFAULT_CONFIG = ROOT / "configs/mappo_actor_lr_retention.yaml"


def validate_config(config, environment_config):
    """Validate the unchanged source setup and three explicit Actor-LR arms."""
    retention._require(isinstance(config.get("retention"), dict), "retention must be a mapping")
    for key in ("curriculum", "warm_start", "continuation", "initial_checkpoint", "resume_checkpoint"):
        retention._require(not config.get(key), "Actor-LR retention cannot enable " + key)
    settings = config["retention"]
    groups = settings.get("groups")
    retention._require(isinstance(groups, list) and len(groups) == 3,
                       "Actor-LR retention requires three A/B/C groups")
    rates = []
    for name, group in zip(("A", "B", "C"), groups):
        retention._require(isinstance(group, dict)
                           and set(group) == {"name", "reset_adam", "actor_learning_rate"}
                           and group["name"] == name and group["reset_adam"] is False,
                           "A/B/C must all retain Adam and specify only their Actor learning rate")
        rate = group["actor_learning_rate"]
        retention._require(not isinstance(rate, bool) and isinstance(rate, (int, float))
                           and np.isfinite(rate) and rate > 0,
                           "Actor learning rates must be finite and positive")
        rates.append(float(rate))
    retention._require(rates[0] == config["ppo"]["actor_learning_rate"]
                       and all(left > right for left, right in zip(rates, rates[1:])),
                       "A must use the source Actor learning rate; B and C must be strictly lower")
    retention._count(settings.get("evaluation_interval_steps"), "retention.evaluation_interval_steps")
    seed = settings.get("evaluation_seed")
    retention._require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
                       "retention.evaluation_seed must be a nonnegative integer")
    retention._count(config["training"].get("total_env_steps"), "training.total_env_steps")
    for key in ("rollout_steps", "log_interval_updates", "checkpoint_interval_updates", "capture_rate_window"):
        retention._count(config["training"].get(key), "training." + key)
    retention._require(environment_config.get("attacker", {}).get("policy", {}).get("mode") == "direct",
                       "Actor-LR retention requires the unchanged Direct Attacker task")
    # Validate the common base PPO mapping against the source first. Per-arm LR
    # is applied only after loading, without clearing Adam moments or counters.
    source_config = deepcopy(config)
    source_config["warm_start"] = {name:settings[name]
                                   for name in ("checkpoint", "expected_sha256") if name in settings}
    return load_warm_start_source(source_config, environment_config)


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
    retention._require(not output.exists(), "Use a new Actor-LR retention output: " + str(output))
    original_run = source["checkpoint"].parent.parent
    retention._require(output != original_run and original_run not in output.parents,
                       "Actor-LR retention output must be independent of the source run")
    config["retention_config_source"] = str(config_path)
    set_seed(config["seed"], config["torch_num_threads"])
    output.mkdir(parents=True)
    write_metadata(output / "metadata.json", config, environment,
        config_path=config_path, env_config_path=env_path, output_dir=output,
        kind="actor_learning_rate_retention_comparison",
        extra=dict(source_checkpoint=str(source["checkpoint"]), source_checkpoint_sha256=source["sha256"],
            source_pretrained_env_steps=source["source_env_steps"],
            treatment="Only Actor optimizer learning rate varies; Actor/Critic weights, both Adam states and source global RNG are inherited.",
            critic_learning_rate=config["ppo"]["critic_learning_rate"],
            learning_rate_scheduler_enabled=False))
    results = [retention.run_group(config, env_path, source, group, output / group["name"])
               for group in config["retention"]["groups"]]
    equal = {}
    with np.load(output / "A/first_rollout.npz", allow_pickle=False) as reference:
        for group in ("B", "C"):
            with np.load(output / group / "first_rollout.npz", allow_pickle=False) as actual:
                retention._require(set(actual.files) == set(reference.files), "First-rollout array schema differs")
                equal[group] = {name:np.array_equal(actual[name], reference[name]) for name in reference.files}
    retention._require(all(all(checks.values()) for checks in equal.values()),
                       "Initial stochastic training data differs across Actor-LR treatments")
    summary = dict(status="completed", output_dir=str(output), groups=results,
        first_rollout_array_equality=equal, source_checkpoint_sha256=source["sha256"],
        source_unchanged=retention._sha(source["checkpoint"]) == source["sha256"],
        actor_learning_rates={group["name"]:group["actor_learning_rate"] for group in config["retention"]["groups"]},
        critic_learning_rate=config["ppo"]["critic_learning_rate"],
        original_adam_states_retained=True, only_actor_learning_rate_varies=True,
        curriculum_training_started=False, learning_rate_scheduler_enabled=False)
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
            critic_learning_rate=config["ppo"]["critic_learning_rate"],
            formal_training_started=False, learning_rate_scheduler_enabled=False), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(run_experiment(args.config, output_dir=args.output_dir, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
