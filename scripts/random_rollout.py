"""Run a reproducible smoke demo: python -m scripts.random_rollout."""

import argparse
import json

import numpy as np

from envs.pursuit_env import DEFAULT_CONFIG, PursuitEnv


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
        terminated = truncated = False
        while not (terminated or truncated):
            observations, rewards, terminated, truncated, info = env.step(env.sample_actions())
            if not (np.isfinite(observations).all() and np.isfinite(rewards).all()
                    and np.isfinite(info["global_state"]).all()):
                raise RuntimeError("Non-finite values encountered during random rollout")
            episode_reward += float(rewards[0])
            min_distance = min(min_distance, info["min_distance"])
            collision_steps += int(info["collision_pairs"] > 0)
        print(json.dumps({
            "episode": episode + 1,
            "seed": episode_seed,
            "steps": env.step_count,
            "captured": terminated,
            "truncated": truncated,
            "episode_reward": episode_reward,
            "min_distance": min_distance,
            "collision_steps": collision_steps,
        }))


if __name__ == "__main__":
    main()
