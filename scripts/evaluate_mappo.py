"""Evaluate a saved MAPPO actor and export metrics and a complete trajectory."""

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import yaml

from utils.evaluation import (
    checkpoint_provenance, load_evaluation_run, run_episode, save_trajectory,
    summarize_episodes, write_json,
)
from utils.experiment import set_seed, write_metadata


def evaluate(checkpoint, episodes=None, seed=None, output_dir=None, env_config=None, device="cpu"):
    agent, payload, env, output_path, environment_source = load_evaluation_run(
        checkpoint, output_dir=output_dir, env_config=env_config, device=device
    )
    config = deepcopy(payload["config"])
    episodes = config["evaluation"]["episodes"] if episodes is None else episodes
    seed = config["evaluation"]["seed"] if seed is None else seed
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("Evaluation episodes must be a positive integer")
    set_seed(seed, config["torch_num_threads"])
    config["seed"] = seed
    config["device"] = device
    config["evaluation"].update(episodes=episodes, seed=seed)
    with (output_path / "evaluation_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    write_metadata(
        output_path / "evaluation_metadata.json", config, env.config,
        config_path=(payload.get("metadata", {}).get("config_source")
                     or output_path / "evaluation_config.yaml"),
        env_config_path=env.config_path, output_dir=output_path, kind="evaluation",
        extra={
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_training_state": checkpoint_provenance(payload),
            "environment_source": environment_source,
            "environment_override": env_config is not None,
            "deterministic": True,
            "episode_seeds": list(range(seed, seed + episodes)),
        },
    )
    rows = []
    for episode in range(episodes):
        episode_seed = seed + episode
        metrics, trajectory = run_episode(env, agent, episode_seed)
        row = {"episode": episode + 1, "seed": episode_seed, **metrics}
        rows.append(row)
        if episode == 0:
            save_trajectory(output_path / "first_episode.npz", trajectory, metrics, episode_seed)
    with (output_path / "evaluation_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_episodes(rows)
    write_json(output_path / "evaluation_summary.json", summary)
    print(json.dumps(summary, allow_nan=False))
    if summary["average_capture_time"] is None:
        print("No interceptions; average_interception_time is null (successful episodes only).")
    print("Deterministic policy with fixed initialization repeats the same episode.")
    print("Evaluation artifacts: {}".format(output_path))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-config", type=Path, help="Explicit environment configuration override")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
