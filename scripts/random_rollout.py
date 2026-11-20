"""Run a reproducible smoke demo: python -m scripts.random_rollout."""

import argparse
import json

import numpy as np

from envs.pursuit_env import DEFAULT_CONFIG, PursuitEnv
from utils.evaluation import episode_outcome, reward_episode_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, help="Override rollout.seed in YAML")
    parser.add_argument("--episodes", type=int, help="Override rollout.episodes in YAML")
    args = parser.parse_args()
    env = PursuitEnv(args.config)
    rollout = env.config["rollout"]
    seed = rollout["seed"] if args.seed is None else args.seed
    episodes = rollout["episodes"] if args.episodes is None else args.episodes
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        parser.error("seed must be a nonnegative integer")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        parser.error("episodes must be a positive integer")
    for episode in range(episodes):
        episode_seed = seed + episode
        observations, info = env.reset(seed=episode_seed)
        episode_reward = 0.0
        min_distance = info["min_distance"]
        collision_steps = 0
        evader_boundary_steps = 0
        consecutive_boundary_steps = 0
        max_consecutive_boundary_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            observations, rewards, terminated, truncated, info = env.step(env.sample_actions())
            if not (np.isfinite(observations).all() and np.isfinite(rewards).all()
                    and np.isfinite(info["global_state"]).all()):
                raise RuntimeError("Non-finite values encountered during random rollout")
            episode_reward += float(rewards[0])
            min_distance = min(min_distance, info["min_distance"])
            collision_steps += int(info["collision_pairs"] > 0)
            # Count exact boundary occupancy separately from actual hard clips:
            # a stationary aircraft can remain on a face without a new clip.
            on_boundary = bool(np.any(
                (env.states[-1, :3] <= env.bounds_low) | (env.states[-1, :3] >= env.bounds_high)
            ))
            evader_boundary_steps += int(on_boundary)
            consecutive_boundary_steps = consecutive_boundary_steps + 1 if on_boundary else 0
            max_consecutive_boundary_steps = max(max_consecutive_boundary_steps, consecutive_boundary_steps)
        print(json.dumps({
            "episode": episode + 1,
            "seed": episode_seed,
            "steps": env.step_count,
            "captured": info["captured"],
            "interception_success": info["interception_success"],
            "attacker_reached_base": info["attacker_reached_base"],
            "enemy_attack_success": info["enemy_attack_success"],
            "interception_failure": info["interception_failure"],
            "outcome": episode_outcome(info, truncated),
            "distance_to_base": info["distance_to_base"],
            "interception_time": env.step_count * env.dt if info["captured"] else None,
            "truncated": truncated,
            "episode_reward": episode_reward,
            "min_distance": min_distance,
            "collision_steps": collision_steps,
            "boundary_steering_enabled": info["boundary_steering_enabled"],
            "boundary_clip_count": info["boundary_clip_count"],
            "boundary_clip_counts": info["boundary_clip_counts"].tolist(),
            "evader_boundary_clip_count": int(info["boundary_clip_counts"][-1]),
            "attacker_boundary_clip_count": int(info["boundary_clip_counts"][-1]),
            "evader_boundary_steps": evader_boundary_steps,
            "evader_max_consecutive_boundary_steps": max_consecutive_boundary_steps,
            **reward_episode_metrics(info),
        }))


if __name__ == "__main__":
    main()
