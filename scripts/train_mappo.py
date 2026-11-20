"""Train the shared-actor MAPPO baseline: python -m scripts.train_mappo."""

import argparse
from collections import deque
import csv
from pathlib import Path

import numpy as np
import yaml

from algorithms.mappo import MAPPOAgent, RolloutBuffer, build_input_scales
from envs import PursuitEnv
from utils.experiment import ROOT, load_training_config, set_seed, write_metadata
from utils.evaluation import REWARD_METRIC_FIELDS, episode_outcome, reward_episode_metrics
from utils.post_training import run_post_training_analysis


DEFAULT_CONFIG = ROOT / "configs" / "mappo_baseline.yaml"

EPISODE_FIELDS = (
    "episode", "env_steps", "episode_reward", "episode_length", "captured",
    "truncated", "capture_rate", "capture_time", "min_distance",
    "final_min_distance", "collision_steps", "collision_step_rate",
    "average_pairwise_pursuer_distance",
    "interception_success", "enemy_attack_success", "attacker_reached_base",
    "interception_failure", "outcome", "distance_to_base", "interception_time",
    "interception_success_rate", "enemy_attack_success_rate", "timeout_rate",
) + REWARD_METRIC_FIELDS
UPDATE_FIELDS = (
    "update", "env_steps", "episodes", "rollout_steps", "capture_rate",
    "mean_episode_reward", "actor_loss", "value_loss", "entropy", "approx_kl",
    "clip_fraction", "actor_grad_norm", "critic_grad_norm",
    "interception_success_rate", "enemy_attack_success_rate", "timeout_rate",
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _episode_start(info):
    return {
        "episode_reward": 0.0,
        "episode_length": 0,
        "min_distance": float(info["min_distance"]),
        "collision_steps": 0,
        "pairwise_distance_sum": 0.0,
        **reward_episode_metrics(info),
    }


def _accumulate_episode(episode, rewards, info, num_agents):
    # Rewards are shared: record one team reward, never their threefold sum.
    reward = float(rewards[0])
    if not np.isfinite(reward):
        raise RuntimeError("Non-finite team reward encountered during training")
    episode["episode_reward"] += reward
    episode["episode_length"] += 1
    episode["min_distance"] = min(episode["min_distance"], float(info["min_distance"]))
    episode["collision_steps"] += int(info["collision_pairs"] > 0)
    positions = np.asarray(info["global_state"]).reshape(num_agents + 1, -1)[:num_agents, :3]
    pairs = np.triu_indices(num_agents, k=1)
    distances = np.linalg.norm(positions[pairs[0]] - positions[pairs[1]], axis=1)
    episode["pairwise_distance_sum"] += float(np.mean(distances))
    episode.update(reward_episode_metrics(info))
    return reward


def train(config_path=DEFAULT_CONFIG, *, total_steps=None, output_dir=None, seed=None):
    """Run sequential environment rollouts and PPO updates; return final counters.

    A rollout may end inside an episode. Episode totals survive PPO updates and
    enter training_metrics.csv only when that episode actually ends. Time-limit
    transitions bootstrap the final state's value before resetting the env.
    """
    config_path = Path(config_path).expanduser().resolve()
    config, env_config_path = load_training_config(config_path)
    if total_steps is not None:
        config["training"]["total_env_steps"] = total_steps
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if seed is not None:
        config["seed"] = seed
    training = config["training"]
    total_steps = _positive_integer(training["total_env_steps"], "training.total_env_steps")
    rollout_steps = _positive_integer(training["rollout_steps"], "training.rollout_steps")
    log_interval = _positive_integer(training["log_interval_updates"], "training.log_interval_updates")
    checkpoint_interval = _positive_integer(
        training["checkpoint_interval_updates"], "training.checkpoint_interval_updates"
    )
    window = _positive_integer(training["capture_rate_window"], "training.capture_rate_window")
    reward_scale = float(config["normalization"]["reward_scale"])
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("normalization.reward_scale must be finite and positive")

    output_path = Path(config["output_dir"]).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.resolve()
    artifacts = ("training_metrics.csv", "updates.csv", "training_config.yaml",
                 "environment_config.yaml", "metadata.json", "checkpoints")
    if output_path.exists() and (not output_path.is_dir()
                                 or any((output_path / name).exists() for name in artifacts)):
        raise ValueError(
            "Training artifacts already exist at: {}. Use --output-dir for a new experiment."
            .format(output_path)
        )
    set_seed(config["seed"], config["torch_num_threads"])
    env = PursuitEnv(env_config_path)
    observations, info = env.reset(seed=config["seed"])
    observation_dim = observations.shape[-1]
    state_dim = info["global_state"].shape[-1]
    action_dim = env.action_shape[-1]
    observation_scale, state_scale = build_input_scales(env.config, config["normalization"])
    agent = MAPPOAgent(
        config, observation_dim=observation_dim, state_dim=state_dim,
        action_dim=action_dim, observation_scale=observation_scale,
        state_scale=state_scale, device=config["device"],
    )
    buffer = RolloutBuffer(
        rollout_steps, env.num_pursuers, observation_dim, state_dim, action_dim
    )

    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    for name, snapshot in (("training_config.yaml", config), ("environment_config.yaml", env.config)):
        with (output_path / name).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(snapshot, stream, sort_keys=False, allow_unicode=True)
    metadata = write_metadata(
        output_path / "metadata.json", config, env.config,
        config_path=config_path, env_config_path=env_config_path,
        output_dir=output_path,
    )
    print("Training config: {}".format(config_path), flush=True)
    print("Environment config: {}".format(env_config_path), flush=True)
    print("Effective configuration:\n{}".format(yaml.safe_dump(config, sort_keys=False)), flush=True)
    print("Output directory: {}".format(output_path), flush=True)

    recent = deque(maxlen=window)
    episode = _episode_start(info)
    counters = {"env_steps": 0, "update": 0, "episodes": 0}
    best_score = None

    def checkpoint_state():
        return dict(
            counters,
            partial_episode=dict(episode),
            recent_episodes=list(recent),
            best_score=best_score,
            best_selection="recent_training_capture_rate_then_mean_reward",
            completed_episodes_available=bool(recent),
            environment_state=env.get_state(),
            environment_accelerations=env.accelerations.copy(),
            environment_step_count=env.step_count,
            environment_rng_state=env.rng.bit_generator.state,
        )

    # Preserve the real untrained policy for later trajectory evolution.
    agent.save(checkpoint_path / "initial.pt", checkpoint_state(), metadata)
    with (output_path / "training_metrics.csv").open("w", newline="", encoding="utf-8") as episodes_file, \
            (output_path / "updates.csv").open("w", newline="", encoding="utf-8") as updates_file:
        episode_writer = csv.DictWriter(episodes_file, fieldnames=EPISODE_FIELDS)
        update_writer = csv.DictWriter(updates_file, fieldnames=UPDATE_FIELDS)
        episode_writer.writeheader()
        update_writer.writeheader()
        while counters["env_steps"] < total_steps:
            buffer.reset()
            completed_this_update = False
            steps_to_collect = min(rollout_steps, total_steps - counters["env_steps"])
            for _ in range(steps_to_collect):
                state = info["global_state"]
                actions, log_probs, raw_actions, value = agent.act(observations, state)
                next_observations, rewards, terminated, truncated, next_info = env.step(actions)
                # Use the final state at timeout; never a state from reset().
                next_value = 0.0 if terminated else agent.value(next_info["global_state"])
                reward = _accumulate_episode(episode, rewards, next_info, env.num_pursuers)
                buffer.add(
                    observations, state, raw_actions, log_probs, reward * reward_scale,
                    value, next_value, terminated, truncated,
                )
                counters["env_steps"] += 1
                observations, info = next_observations, next_info
                if terminated or truncated:
                    counters["episodes"] += 1
                    captured = bool(info["captured"])
                    enemy_success = bool(info.get("enemy_attack_success", False))
                    recent.append({
                        "captured": int(captured), "enemy_attack_success": int(enemy_success),
                        "truncated": int(truncated), "episode_reward": episode["episode_reward"],
                    })
                    capture_rate = float(np.mean([entry["captured"] for entry in recent]))
                    enemy_rate = float(np.mean([entry["enemy_attack_success"] for entry in recent]))
                    timeout_rate = float(np.mean([entry["truncated"] for entry in recent]))
                    length = episode["episode_length"]
                    episode_writer.writerow({
                        "episode": counters["episodes"], "env_steps": counters["env_steps"],
                        "episode_reward": episode["episode_reward"], "episode_length": length,
                        "captured": int(captured), "truncated": int(truncated),
                        "capture_rate": capture_rate,
                        "capture_time": length * env.dt if captured else None,
                        "interception_success": int(captured),
                        "enemy_attack_success": int(enemy_success),
                        "attacker_reached_base": int(info.get("attacker_reached_base", False)),
                        "interception_failure": int(info.get("interception_failure", False)),
                        "outcome": episode_outcome(info, truncated),
                        "distance_to_base": info.get("distance_to_base"),
                        "interception_time": length * env.dt if captured else None,
                        "interception_success_rate": capture_rate,
                        "enemy_attack_success_rate": enemy_rate,
                        "timeout_rate": timeout_rate,
                        "min_distance": episode["min_distance"],
                        "final_min_distance": info["min_distance"],
                        "collision_steps": episode["collision_steps"],
                        "collision_step_rate": episode["collision_steps"] / length,
                        "average_pairwise_pursuer_distance": episode["pairwise_distance_sum"] / length,
                        **{name: episode[name] for name in REWARD_METRIC_FIELDS},
                    })
                    episodes_file.flush()
                    completed_this_update = True
                    observations, info = env.reset()
                    episode = _episode_start(info)

            buffer.compute_returns(config["ppo"]["gamma"], config["ppo"]["gae_lambda"])
            losses = agent.update(buffer)
            if not all(np.isfinite(float(value)) for value in losses.values()):
                raise RuntimeError("PPO update returned a non-finite metric")
            counters["update"] += 1
            capture_rate = float(np.mean([entry["captured"] for entry in recent])) if recent else None
            enemy_rate = float(np.mean([entry["enemy_attack_success"] for entry in recent])) if recent else None
            timeout_rate = float(np.mean([entry["truncated"] for entry in recent])) if recent else None
            mean_reward = float(np.mean([entry["episode_reward"] for entry in recent])) if recent else None
            row = dict(counters, rollout_steps=buffer.size, capture_rate=capture_rate,
                       interception_success_rate=capture_rate,
                       enemy_attack_success_rate=enemy_rate, timeout_rate=timeout_rate,
                       mean_episode_reward=mean_reward, **losses)
            update_writer.writerow(row)
            updates_file.flush()

            if completed_this_update:
                score = (capture_rate, mean_reward)
                if best_score is None or score > best_score:
                    best_score = score
                    agent.save(checkpoint_path / "best.pt", checkpoint_state(), metadata)
            if counters["update"] % checkpoint_interval == 0:
                agent.save(
                    checkpoint_path / "update_{:06d}.pt".format(counters["update"]),
                    checkpoint_state(), metadata,
                )
            if counters["update"] % log_interval == 0 or counters["env_steps"] == total_steps:
                rate_text = "n/a" if capture_rate is None else "{:.3f}".format(capture_rate)
                reward_text = "n/a" if mean_reward is None else "{:.3f}".format(mean_reward)
                print(
                    "update={} steps={}/{} episodes={} interception_rate={} enemy_attack_rate={} "
                    "timeout_rate={} mean_reward={} "
                    "actor_loss={:.4f} value_loss={:.4f} entropy={:.4f}".format(
                        counters["update"], counters["env_steps"], total_steps, counters["episodes"],
                        rate_text, "n/a" if enemy_rate is None else "{:.3f}".format(enemy_rate),
                        "n/a" if timeout_rate is None else "{:.3f}".format(timeout_rate),
                        reward_text, losses["actor_loss"], losses["value_loss"], losses["entropy"],
                    ), flush=True,
                )

    final_state = checkpoint_state()
    agent.save(checkpoint_path / "final.pt", final_state, metadata)
    if best_score is None:
        # An extremely short smoke run may contain no complete episode.
        agent.save(checkpoint_path / "best.pt", final_state, metadata)
        print("No completed episodes: best.pt contains the final smoke-run policy.", flush=True)
    summary = dict(counters, output_dir=str(output_path), final_checkpoint=str(checkpoint_path / "final.pt"))
    print("Training completed: {}".format(summary), flush=True)
    run_post_training_analysis(output_path, config)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="MAPPO training YAML")
    parser.add_argument("--total-steps", type=int, help="Override training.total_env_steps")
    parser.add_argument("--output-dir", help="Directory without existing training artifacts")
    parser.add_argument("--seed", type=int, help="Override the training random seed")
    args = parser.parse_args()
    try:
        train(args.config, total_steps=args.total_steps, output_dir=args.output_dir, seed=args.seed)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
