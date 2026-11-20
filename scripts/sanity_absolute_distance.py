"""Verify distance-time rewards using complete stationary/random episodes."""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from envs import PursuitEnv
from scripts.sanity_reward import SanityPolicy
from utils.evaluation import run_episode, save_trajectory, summarize_episodes, write_json
from utils.experiment import ROOT


def audit(trajectory, metrics, config):
    for value in trajectory.values():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RuntimeError("Non-finite trajectory")
    positions = trajectory["positions"]
    distance = np.linalg.norm(positions[:, :3] - positions[:, 3, None], axis=-1).min(axis=1)
    dt = float(config["environment"]["dt"])
    reward = config["reward"]
    integral = distance[1:] * dt
    costs = -reward["distance_cost_coefficient"] * integral
    names = trajectory["reward_component_names"].tolist()
    parts = dict(zip(names, trajectory["reward_components"].T))
    np.testing.assert_allclose(parts["distance_to_nearest_pursuer"], distance[1:], atol=1e-10)
    np.testing.assert_allclose(parts["distance_time_integral"], integral, atol=1e-10)
    np.testing.assert_allclose(parts["absolute_distance_cost"], costs, atol=1e-10)
    assert "active_progress_reward" not in parts and "progress" not in parts
    safety = config["safety"]["pursuer_safety_distance"]
    pair_distances = np.stack([np.linalg.norm(positions[1:, i] - positions[1:, j], axis=-1)
                               for i, j in ((0, 1), (0, 2), (1, 2))], axis=1)
    collision = np.where((pair_distances < safety).any(axis=1), reward["collision_penalty"], 0.0)
    survival = np.full(len(costs), reward["survival_penalty_per_second"] * dt)
    capture, failure = np.zeros_like(costs), np.zeros_like(costs)
    if metrics["captured"]:
        capture[-1] = reward["capture_reward"]
    elif metrics["enemy_attack_success"]:
        failure[-1] = reward["attack_failure_penalty"]
    for name, expected in (("collision", collision), ("survival", survival),
                           ("capture", capture), ("attack_failure", failure)):
        np.testing.assert_allclose(parts[name], expected, atol=1e-10)
    expected = costs + collision + survival + capture + failure
    np.testing.assert_allclose(trajectory["rewards"], expected, atol=1e-10)
    np.testing.assert_allclose(metrics["episode_reward"], expected.sum(), atol=1e-9)
    np.testing.assert_allclose(metrics["cumulative_absolute_distance_cost"], costs.sum(), atol=1e-9)
    assert metrics["cumulative_active_progress_reward"] is None
    if metrics["enemy_attack_success"]:
        assert metrics["episode_reward"] < 0
    return {"post_step_distance_time_integral_m_s": float(integral.sum()),
            "distance_cost": float(costs.sum()), "weighted_reward_reconstructed": True,
            "finite": True}


def check(config_path, output_dir, episodes=10, seed=0, reference_sanity=None):
    if episodes < 10:
        raise ValueError("At least 10 complete random episodes are required")
    env = PursuitEnv(config_path)
    if not env.absolute_distance_enabled:
        raise ValueError("Expected the absolute-distance schema")
    output = Path(output_dir).resolve()
    assert not (output / "reward_sanity_summary.json").exists()
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment_config.yaml").write_text(yaml.safe_dump(env.config, sort_keys=False), encoding="utf-8")
    if reference_sanity is not None:
        reference_sanity = Path(reference_sanity)
        reference_config = yaml.safe_load((reference_sanity / "environment_config.yaml").read_text(encoding="utf-8"))
        assert {k: v for k, v in env.config.items() if k != "reward"} == {
            k: v for k, v in reference_config.items() if k != "reward"}
    rows, audits = [], []
    reference_matches = []
    stationary, trajectory = run_episode(env, SanityPolicy(env), seed)
    assert np.count_nonzero(trajectory["states"][:, :3, 3:]) == 0
    stationary_audit = audit(trajectory, stationary, env.config)
    if reference_sanity is not None:
        with np.load(reference_sanity / "stationary_episode.npz", allow_pickle=False) as reference:
            np.testing.assert_array_equal(trajectory["states"], reference["states"])
        reference_matches.append("stationary")
    save_trajectory(output / "stationary_episode.npz", trajectory, stationary, seed)
    write_json(output / "stationary_metrics.json", stationary)
    random_dir = output / "random"
    random_dir.mkdir(exist_ok=True)
    for episode in range(episodes):
        metrics, trajectory = run_episode(env, SanityPolicy(env, random_actions=True), seed + episode)
        audits.append(audit(trajectory, metrics, env.config))
        filename = "episode_{:03d}.npz".format(episode + 1)
        if reference_sanity is not None:
            with np.load(reference_sanity / "random" / filename, allow_pickle=False) as reference:
                np.testing.assert_array_equal(trajectory["states"], reference["states"])
            reference_matches.append(filename)
        rows.append({"episode": episode + 1, "seed": seed + episode, **metrics})
        save_trajectory(random_dir / filename, trajectory, metrics, seed + episode)
    with (output / "random_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"sanity_passed": True, "coefficient": env.reward_coefficients["distance_cost_coefficient"],
               "seed": seed, "random_seeds": list(range(seed, seed + episodes)),
               "stationary": stationary, "stationary_audit": stationary_audit,
               "random": summarize_episodes(rows), "random_audits": audits,
               "historical_state_trajectories_bit_identical": reference_matches,
               "note": "Same physical scenarios and actions; only the reward changes. No generalization claim."}
    write_json(output / "reward_sanity_summary.json", summary)
    print("Stationary reward={:.6f}; random mean={:.6f}; random capture={:.3f}".format(
        stationary["episode_reward"], summary["random"]["average_episode_reward"], summary["random"]["capture_rate"]))
    print("Sanity passed; {} old physical trajectories match exactly.".format(len(reference_matches)))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", type=Path, default=ROOT / "configs/baseline_absolute_distance.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-sanity", type=Path)
    args = parser.parse_args()
    check(**vars(args))
