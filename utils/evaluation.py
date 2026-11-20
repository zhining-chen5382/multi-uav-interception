"""Deterministic rollouts and portable artifacts for MAPPO evaluation."""

import json
from pathlib import Path

import numpy as np
import yaml

from envs.pursuit_env import PursuitEnv


REWARD_METRIC_FIELDS = (
    "cumulative_active_progress_reward", "cumulative_survival_penalty",
    "cumulative_collision_penalty", "terminal_reward",
    "cumulative_absolute_distance_cost",
)


def reward_episode_metrics(info):
    """Expose weighted episode totals; absent historical terms remain null.

    Raw distance differences in reward_components are diagnostics, not reward
    terms, and must never be added to the episode return a second time.
    """
    components = info.get("episode_reward_components", {})
    return {
        "cumulative_active_progress_reward": components.get("active_progress_reward"),
        "cumulative_survival_penalty": components.get("survival"),
        "cumulative_collision_penalty": components.get("collision"),
        "terminal_reward": (components["capture"] + components.get("attack_failure", 0.0)
                            if "capture" in components else None),
        "cumulative_absolute_distance_cost": components.get("absolute_distance_cost"),
    }


def run_episode(env, agent, seed):
    """Execute the shared actor using only each pursuer's local observation.

    Team reward is counted once per step. Minimum distance includes the reset
    state. Mean pairwise distance and collision steps use post-step states,
    matching training metrics and the environment's collision penalty.
    """
    observations, info = env.reset(seed=seed)
    states = [env.states.copy()]
    min_distances = [float(info["min_distance"])]
    collision_pairs = [int(info["collision_pairs"])]
    base_distances = [info.get("distance_to_base")]
    actions_history = []
    rewards_history = []
    component_names = None
    component_history = []
    terminated = truncated = False
    while not (terminated or truncated):
        # No centralized state reaches the actor during evaluation (CTDE).
        actions, _, _, _ = agent.act(observations, deterministic=True)
        observations, rewards, terminated, truncated, info = env.step(actions)
        states.append(env.states.copy())
        actions_history.append(np.asarray(actions).copy())
        rewards_history.append(float(rewards[0]))
        components = info.get("reward_components", {})
        if component_names is None:
            component_names = tuple(components)
        component_history.append([components[name] for name in component_names])
        min_distances.append(float(info["min_distance"]))
        collision_pairs.append(int(info["collision_pairs"]))
        base_distances.append(info.get("distance_to_base"))

    states = np.asarray(states)
    positions = states[:, :, :3]
    pairwise_distances = np.stack([
        np.linalg.norm(positions[:, i] - positions[:, j], axis=-1)
        for i in range(env.num_pursuers) for j in range(i + 1, env.num_pursuers)
    ], axis=1)
    length = int(env.step_count)
    collision_steps = int(np.count_nonzero(collision_pairs[1:]))
    metrics = {
        "captured": bool(info["captured"]),
        "interception_success": bool(info.get("interception_success", info["captured"])),
        "attacker_reached_base": bool(info.get("attacker_reached_base", False)),
        "enemy_attack_success": bool(info.get("enemy_attack_success", False)),
        "interception_failure": bool(info.get("interception_failure", False)),
        "truncated": bool(truncated),
        "outcome": episode_outcome(info, truncated),
        "mission_enabled": bool(getattr(env, "mission_enabled", False)),
        "episode_reward": float(np.sum(rewards_history)),
        "episode_length": length,
        "capture_time": float(length * env.dt) if info["captured"] else None,
        "interception_time": float(length * env.dt) if info["captured"] else None,
        "attacker_base_terminal_distance": info.get("distance_to_base"),
        "min_distance": float(np.min(min_distances)),
        "final_min_distance": float(info["min_distance"]),
        "collision_steps": collision_steps,
        "collision_step_rate": float(collision_steps / length),
        "average_pairwise_pursuer_distance": float(np.mean(pairwise_distances[1:])),
        "boundary_clip_count": int(info.get("boundary_clip_count", 0)),
        "attacker_boundary_clip_count": int(info["boundary_clip_counts"][-1])
            if "boundary_clip_counts" in info else 0,
        **reward_episode_metrics(info),
    }
    trajectory = {
        "positions": positions,
        "states": states,
        "time": np.arange(length + 1, dtype=np.float64) * env.dt,
        "actions": np.asarray(actions_history),
        "rewards": np.asarray(rewards_history),
        "min_distance": np.asarray(min_distances),
        "collision_pairs": np.asarray(collision_pairs, dtype=np.int64),
        # One row per transition; positions/time also include the reset state.
        "reward_component_names": np.asarray(component_names, dtype=str),
        "reward_components": np.asarray(component_history, dtype=np.float64),
    }
    if getattr(env, "mission_enabled", False):
        trajectory.update(
            base_position=np.asarray(env.base_position).copy(),
            attack_radius=np.asarray(env.attack_radius),
            distance_to_base=np.asarray(base_distances, dtype=np.float64),
        )
    return metrics, trajectory


def episode_outcome(info, truncated):
    """Distinguish task outcomes without confusing enemy victory with capture."""
    if info.get("interception_success", info["captured"]):
        return "Interception Success"
    if info.get("attacker_reached_base", False):
        return "Enemy Attack Success"
    if truncated:
        return "Timeout"
    return "Ongoing"


def load_evaluation_run(checkpoint, output_dir=None, env_config=None, device="cpu",
                        artifact_prefix="evaluation"):
    """Load a checkpoint and recreate its snapshotted training environment.

    An environment override is only applied when explicitly provided. The
    saved environment YAML has a distinct name to preserve training artifacts.
    """
    from algorithms.mappo import MAPPOAgent

    checkpoint = Path(checkpoint).resolve()
    agent, payload = MAPPOAgent.from_checkpoint(checkpoint, device=device)
    output_path = (
        Path(output_dir).resolve() if output_dir is not None
        else checkpoint.parent.parent
    )
    output_path.mkdir(parents=True, exist_ok=True)
    if env_config is not None:
        source = str(Path(env_config).resolve())
        with Path(source).open(encoding="utf-8") as stream:
            environment_config = yaml.safe_load(stream)
    else:
        environment_config = payload.get("metadata", {}).get("environment_config")
        if environment_config is None:
            raise ValueError(
                "Checkpoint has no environment_config snapshot; supply --env-config explicitly"
            )
        source = "checkpoint:metadata.environment_config"
    if not isinstance(environment_config, dict):
        raise ValueError("Environment configuration must be a YAML mapping")
    snapshot_path = output_path / (artifact_prefix + "_environment_config.yaml")
    with snapshot_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(environment_config, stream, sort_keys=False, allow_unicode=True)
    env = PursuitEnv(snapshot_path)
    agent.actor.eval()
    agent.critic.eval()
    return agent, payload, env, output_path, source


def save_trajectory(path, trajectory, metrics, seed):
    """Save numeric arrays and JSON text, so loading never requires pickle."""
    metadata = json.dumps({"seed": seed, "metrics": metrics}, allow_nan=False)
    np.savez_compressed(path, metadata_json=np.asarray(metadata), **trajectory)


def checkpoint_provenance(payload):
    """Keep JSON counters and selection details; full simulator state stays in PT."""
    state = payload.get("training_state", {})
    provenance = {}
    for name in ("env_steps", "update", "episodes", "best_selection",
                 "completed_episodes_available"):
        if name not in state:
            continue
        value = state[name]
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int)):
            provenance[name] = value
        elif isinstance(value, float) and np.isfinite(value):
            provenance[name] = value
    return provenance


def summarize_episodes(rows):
    """Aggregate episode metrics; unsuccessful capture time remains JSON null."""
    if not rows:
        raise ValueError("At least one evaluation episode is required")
    capture_times = [row["capture_time"] for row in rows if row["captured"]]
    base_distances = [row["attacker_base_terminal_distance"] for row in rows
                      if row.get("attacker_base_terminal_distance") is not None]
    collision_steps = sum(row["collision_steps"] for row in rows)
    total_steps = sum(row["episode_length"] for row in rows)
    summary = {
        "episodes": len(rows),
        "capture_rate": float(np.mean([row["captured"] for row in rows])),
        "interception_success_rate": float(np.mean([
            row.get("interception_success", row["captured"]) for row in rows
        ])),
        "enemy_attack_success_rate": float(np.mean([
            row.get("enemy_attack_success", False) for row in rows
        ])),
        "timeout_rate": float(np.mean([
            row.get("truncated", not row["captured"]
                    and not row.get("enemy_attack_success", False)) for row in rows
        ])),
        "average_attacker_base_terminal_distance": (
            float(np.mean(base_distances)) if base_distances else None
        ),
        "average_episode_reward": float(np.mean([row["episode_reward"] for row in rows])),
        "average_capture_time": float(np.mean(capture_times)) if capture_times else None,
        "average_interception_time": float(np.mean(capture_times)) if capture_times else None,
        "average_episode_length": float(np.mean([row["episode_length"] for row in rows])),
        "average_min_distance": float(np.mean([row["min_distance"] for row in rows])),
        "average_final_min_distance": float(np.mean([row["final_min_distance"] for row in rows])),
        "collision_episode_rate": float(np.mean([row["collision_steps"] > 0 for row in rows])),
        "collision_steps": int(collision_steps),
        "collision_step_rate": float(collision_steps / total_steps),
        "average_pairwise_pursuer_distance": float(np.mean([
            row["average_pairwise_pursuer_distance"] for row in rows
        ])),
    }
    for name in REWARD_METRIC_FIELDS:
        values = [row.get(name) for row in rows]
        summary["average_" + name] = (
            float(np.mean(values)) if all(value is not None for value in values) else None
        )
    return summary


def write_json(path, data):
    """Reject non-finite values instead of emitting nonstandard JSON NaN."""
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
