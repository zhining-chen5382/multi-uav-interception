"""Check active-progress rewards with stationary and uniform-random pursuers."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml

from envs.pursuit_env import DEFAULT_CONFIG, PursuitEnv
from utils.evaluation import run_episode, save_trajectory, summarize_episodes, write_json


class SanityPolicy:
    """Evaluation adapter; zero acceleration preserves the fixed zero velocity."""

    def __init__(self, env, random_actions=False):
        self.env = env
        self.random_actions = random_actions

    def act(self, observations, deterministic=True):
        actions = (self.env.sample_actions() if self.random_actions
                   else np.zeros(self.env.action_shape, dtype=np.float64))
        return actions, None, None, None


def validate_reward_trajectory(trajectory, metrics, config):
    """Independently reconstruct distance differences and weighted returns."""
    for values in trajectory.values():
        if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
            raise RuntimeError("Non-finite trajectory values")
    names = list(trajectory["reward_component_names"])
    components = dict(zip(names, trajectory["reward_components"].T))
    positions = trajectory["positions"]
    nearest = np.linalg.norm(positions[:, :3] - positions[:, 3, None], axis=-1).min(axis=1)
    base = np.asarray(config["mission"]["base_position"])
    base_distance = np.linalg.norm(positions[:, 3] - base, axis=1)
    pe_progress, eb_progress = -np.diff(nearest), -np.diff(base_distance)
    coefficients = config["reward"]
    dt = config["environment"]["dt"]
    np.testing.assert_allclose(components["pursuer_attacker_progress"], pe_progress, atol=1e-10)
    np.testing.assert_allclose(components["attacker_base_progress"], eb_progress, atol=1e-10)
    np.testing.assert_allclose(components["active_progress"], pe_progress - eb_progress, atol=1e-10)
    np.testing.assert_allclose(components["active_progress_reward"],
                               coefficients["active_progress_coefficient"] * (pe_progress - eb_progress),
                               atol=1e-10)
    np.testing.assert_allclose(components["survival"], coefficients["survival_penalty_per_second"] * dt)
    weighted_names = ("active_progress_reward", "survival", "collision", "capture", "attack_failure")
    np.testing.assert_allclose(trajectory["rewards"], sum(components[name] for name in weighted_names),
                               atol=1e-10)
    np.testing.assert_allclose(sum(trajectory["rewards"]), metrics["episode_reward"], atol=1e-9)
    expected = {
        "cumulative_active_progress_reward": components["active_progress_reward"].sum(),
        "cumulative_survival_penalty": components["survival"].sum(),
        "cumulative_collision_penalty": components["collision"].sum(),
        "terminal_reward": (components["capture"] + components["attack_failure"]).sum(),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(metrics[name], value, atol=1e-9)
    return {
        "finite": True, "reward_reconstructed": True,
        "pursuer_attacker_progress": float(pe_progress.sum()),
        "attacker_base_progress": float(eb_progress.sum()),
        "active_progress": float((pe_progress - eb_progress).sum()),
    }


def check(config_path=DEFAULT_CONFIG, output_dir="outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity",
          episodes=10, seed=None):
    if episodes < 10:
        raise ValueError("Reward sanity requires at least 10 random episodes")
    env = PursuitEnv(config_path)
    if "active_progress_coefficient" not in env.reward_coefficients:
        raise ValueError("Reward sanity requires the active-progress reward schema")
    seed = env.config["rollout"]["seed"] if seed is None else seed
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output_dir).resolve()
    if (output / "reward_sanity_summary.json").exists():
        raise FileExistsError("Choose a new sanity output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "environment_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(env.config, stream, sort_keys=False)
    if np.any(env._initial_states[:3, 3:] != 0.0):
        raise ValueError("Stationary test requires the existing zero-velocity initialization")
    stationary, trajectory = run_episode(env, SanityPolicy(env), seed)
    initial_pursuers = trajectory["states"][0, :3]
    np.testing.assert_array_equal(trajectory["states"][:, :3],
                                  np.broadcast_to(initial_pursuers, trajectory["states"][:, :3].shape))
    stationary_audit = validate_reward_trajectory(trajectory, stationary, env.config)
    save_trajectory(output / "stationary_episode.npz", trajectory, stationary, seed)
    write_json(output / "stationary_metrics.json", stationary)
    rows, audits = [], []
    random_dir = output / "random"
    random_dir.mkdir(exist_ok=True)
    for episode in range(episodes):
        episode_seed = seed + episode
        metrics, trajectory = run_episode(env, SanityPolicy(env, random_actions=True), episode_seed)
        audits.append(validate_reward_trajectory(trajectory, metrics, env.config))
        rows.append(dict(episode=episode + 1, seed=episode_seed, **metrics))
        save_trajectory(random_dir / "episode_{:03d}.npz".format(episode + 1),
                        trajectory, metrics, episode_seed)
    with (output / "random_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    random_summary = summarize_episodes(rows)
    failed = [row for row in rows if row["enemy_attack_success"]]
    all_failed_negative = all(row["episode_reward"] < 0.0 for row in failed) if failed else None
    tolerance = np.finfo(np.float64).eps * env.detection_radius * env.max_episode_steps
    stationary_passed = (stationary["enemy_attack_success"] and stationary["episode_reward"] < 0.0
                         and stationary["cumulative_active_progress_reward"] <= tolerance)
    summary = {
        "config_source": str(env.config_path),
        "config_sha256": hashlib.sha256(env.config_path.read_bytes()).hexdigest(),
        "seed": seed, "random_episode_seeds": list(range(seed, seed + episodes)),
        "stationary": stationary, "stationary_audit": stationary_audit,
        "random": random_summary, "random_audits": audits,
        "random_failure_episodes": len(failed),
        "random_failure_rewards_all_negative": all_failed_negative,
        "random_failure_reward_range": [min(row["episode_reward"] for row in failed),
                                         max(row["episode_reward"] for row in failed)] if failed else None,
        "sanity_passed": bool(stationary_passed and all_failed_negative is not False),
        "note": "This check concerns the fixed configured task; no universal reward-sign guarantee is inferred.",
    }
    write_json(output / "reward_sanity_summary.json", summary)
    print("Stationary reward={:.6f}; active={:.6f}; enemy_success={}".format(
        stationary["episode_reward"], stationary["cumulative_active_progress_reward"],
        stationary["enemy_attack_success"]), flush=True)
    print("Random {} episodes: reward={:.6f}; interception={:.3f}; enemy_success={:.3f}".format(
        episodes, random_summary["average_episode_reward"], random_summary["interception_success_rate"],
        random_summary["enemy_attack_success_rate"]), flush=True)
    print("Reward sanity passed: {} | {}".format(summary["sanity_passed"], output), flush=True)
    if not summary["sanity_passed"]:
        raise RuntimeError("Reward sanity failed; inspect actual results before training")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    check(args.config, args.output_dir, args.episodes, args.seed)


if __name__ == "__main__":
    main()
