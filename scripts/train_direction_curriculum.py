"""v09 arrival-direction curriculum; the original v08 trainer is unchanged."""

import argparse
from collections import deque
from copy import deepcopy
import csv
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from algorithms.mappo import MAPPOAgent, RolloutBuffer, build_input_scales
from scripts.train_mappo import EPISODE_FIELDS, UPDATE_FIELDS, _accumulate_episode, _episode_start
from utils.direction_curriculum import (
    DirectionCurriculumEnv, PromotionGate, arrival_position, evaluation_angles,
    evaluation_key, load_best_eval_checkpoint, restore_rng_state, stage_evaluation_settings,
)
from utils.curriculum_resume import (
    copy_resume_artifacts, load_resume_source, restore_environment, snapshot_environment,
)
from utils.curriculum_warm_start import (
    EVALUATION_TIMINGS, evaluation_due, load_warm_start_source, rollout_collection_steps,
)
from utils.evaluation import REWARD_METRIC_FIELDS, episode_outcome, run_episode, save_trajectory, summarize_episodes, write_json
from utils.experiment import ROOT, load_training_config, set_seed, write_metadata
from utils.post_training import run_post_training_analysis


DEFAULT_CONFIG = ROOT / "configs/mappo_direction_curriculum_zero_start.yaml"
EVALUATION_FIELDS = (
    "stage_index", "half_angle_degrees", "stage_env_steps", "env_steps", "update", "role",
    "evaluated_half_angle_degrees", "capture_rate", "enemy_attack_success_rate", "average_capture_time",
    "average_min_distance", "collision_episode_rate", "passed", "consecutive_passes", "checkpoint",
)


def validate_config(config):
    curriculum = config["curriculum"]
    angles = curriculum["stage_half_angles_degrees"]
    if not isinstance(angles, list) or not angles or any(
        isinstance(v, bool) or not np.isfinite(v) or not 0 <= v <= 180 for v in angles
    ) or any(a >= b for a, b in zip(angles, angles[1:])):
        raise ValueError("Stage half angles must be finite, nonnegative and strictly increasing")
    positive = ("max_stage_steps", "evaluation_interval_steps", "evaluation_episodes",
                "required_consecutive_evaluations", "stage_sampling_seed_stride")
    for name in positive:
        value = curriculum[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("curriculum.{} must be a positive integer".format(name))
    if curriculum["evaluation_episodes"] < 2:
        raise ValueError("At least two fixed evaluation angles are required")
    for half_angle in angles:
        stage_evaluation_settings(curriculum, half_angle)
    budget, interval = curriculum["max_stage_steps"], curriculum["evaluation_interval_steps"]
    budgets = curriculum.get("stage_step_budgets", [budget] * len(angles))
    if (not isinstance(budgets, list) or len(budgets) != len(angles)
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % interval for v in budgets)):
        raise ValueError("Every stage budget must be a positive multiple of the evaluation interval")
    total = config["training"]["total_env_steps"]
    if (isinstance(total, bool) or not isinstance(total, int) or total <= 0 or total % interval
            or total > sum(budgets) or budget % interval
            or ("stage_step_budgets" not in curriculum and total != sum(budgets))):
        raise ValueError("Total budget must be an evaluation multiple within the stage caps")
    rate = curriculum["promotion_capture_rate"]
    if isinstance(rate, bool) or not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("promotion_capture_rate must lie in [0,1]")
    if curriculum["collision_selection_metric"] != "collision_episode_rate":
        raise ValueError("The v09 final tie-breaker is collision_episode_rate")
    if not isinstance(curriculum["monitor_previous_stage"], bool):
        raise ValueError("monitor_previous_stage must be boolean")
    for name in ("rollout_steps", "log_interval_updates", "checkpoint_interval_updates", "capture_rate_window"):
        value = config["training"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("training.{} must be a positive integer".format(name))
    timing = curriculum.get("evaluation_timing", "exact_steps")
    if timing not in EVALUATION_TIMINGS:
        raise ValueError("curriculum.evaluation_timing must be exact_steps or after_full_rollout")
    if timing == "after_full_rollout" and interval < config["training"]["rollout_steps"]:
        raise ValueError("Full-rollout evaluation interval must be at least one rollout")
    if "warm_start" in config:
        warm = config["warm_start"]
        if not isinstance(warm, dict) or not warm or not isinstance(warm.get("checkpoint"), str) or not warm["checkpoint"].strip():
            raise ValueError("warm_start must provide a checkpoint path")
        if config.get("continuation"):
            raise ValueError("warm_start and same-run continuation cannot be combined")
        previous = warm.get("previous_stage_half_angle_degrees")
        if previous is not None:
            if (isinstance(previous, bool) or not isinstance(previous, (int, float))
                    or not np.isfinite(previous) or not 0 <= previous < angles[0]):
                raise ValueError("warm_start.previous_stage_half_angle_degrees must precede the first stage")
            stage_evaluation_settings(curriculum, previous)
    for name, value in (("seed", config["seed"]), ("evaluation_seed", curriculum["evaluation_seed"])):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    scale = config["normalization"]["reward_scale"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reward_scale must be finite and positive")
    if config.get("resume_checkpoint") or config.get("initial_checkpoint"):
        raise ValueError("Stage 1 starts from scratch; no external initial checkpoint is accepted")
    return curriculum


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def evaluate_policy(agent, env_config_path, half_angle_degrees, episodes, seed, output_dir):
    """Evaluate an endpoint-inclusive fixed grid without disturbing training."""
    output = Path(output_dir)
    if (output / "evaluation_metrics.csv").exists() or (output / "summary.json").exists():
        raise ValueError("Evaluation artifacts already exist: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    before_rng = _rng_state()
    modes = (agent.actor.training, agent.critic.training)
    rows = []
    angles = evaluation_angles(half_angle_degrees, episodes)
    try:
        agent.actor.eval()
        agent.critic.eval()
        env = DirectionCurriculumEnv(env_config_path, half_angle_degrees, seed, fixed_angle_degrees=float(angles[0]))
        with torch.inference_mode():
            for index, angle in enumerate(angles):
                env.fixed_angle_degrees = float(angle)
                metrics, trajectory = run_episode(env, agent, seed + index)
                position = trajectory["positions"][0, -1]
                rows.append(dict(episode=index + 1, seed=seed + index, angle_degrees=float(angle),
                                 initial_attacker_x=float(position[0]), initial_attacker_y=float(position[1]),
                                 initial_attacker_z=float(position[2]), **metrics))
                if index == 0:
                    save_trajectory(output / "first_episode.npz", trajectory, metrics, seed)
        with (output / "evaluation_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize_episodes(rows)
        write_json(output / "summary.json", summary)
        write_json(output / "angles.json", angles.tolist())
        return summary
    finally:
        restore_rng_state(before_rng)
        agent.actor.train(modes[0])
        agent.critic.train(modes[1])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_checkpoint(source, destination):
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


def _episode_row(episode, info, truncated, recent, counters, env):
    length = episode["episode_length"]
    capture_rate = float(np.mean([r["captured"] for r in recent]))
    enemy_rate = float(np.mean([r["enemy_attack_success"] for r in recent]))
    timeout_rate = float(np.mean([r["truncated"] for r in recent]))
    captured = bool(info["captured"])
    return dict(
        episode=counters["episodes"], env_steps=counters["env_steps"],
        episode_reward=episode["episode_reward"], episode_length=length,
        captured=int(captured), truncated=int(truncated), capture_rate=capture_rate,
        capture_time=length * env.dt if captured else None,
        min_distance=episode["min_distance"], final_min_distance=info["min_distance"],
        collision_steps=episode["collision_steps"], collision_step_rate=episode["collision_steps"] / length,
        average_pairwise_pursuer_distance=episode["pairwise_distance_sum"] / length,
        interception_success=int(captured), enemy_attack_success=int(info["enemy_attack_success"]),
        attacker_reached_base=int(info["attacker_reached_base"]), interception_failure=int(info["interception_failure"]),
        outcome=episode_outcome(info, truncated), distance_to_base=info["distance_to_base"],
        interception_time=length * env.dt if captured else None,
        interception_success_rate=capture_rate, enemy_attack_success_rate=enemy_rate, timeout_rate=timeout_rate,
        **{name: episode[name] for name in REWARD_METRIC_FIELDS}
    )


def train_curriculum(config_path=DEFAULT_CONFIG, output_dir=None, evaluation_fn=None, seed=None):
    """Train the configured first stage fresh, then transfer its best evaluation.

    env_steps counts all actual work, including updates later discarded when
    selecting an earlier best checkpoint. It is never rolled back on transfer.
    """
    config_path = Path(config_path).resolve()
    config, env_config_path = load_training_config(config_path)
    if seed is not None:
        config["seed"] = seed
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    curriculum = validate_config(config)
    output = Path(config["output_dir"]).expanduser()
    output = (output if output.is_absolute() else ROOT / output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new empty curriculum output directory: " + str(output))
    evaluator = evaluate_policy if evaluation_fn is None else evaluation_fn
    environment_config = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
    if environment_config.get("attacker", {}).get("policy", {}).get("mode") != "direct":
        raise ValueError("Direction curriculum requires the unchanged Direct Attacker")
    warm = load_warm_start_source(config, environment_config) if "warm_start" in config else None
    continuation = config.get("continuation", {})
    if not isinstance(continuation, dict):
        raise ValueError("continuation must be a mapping")
    resume = None
    if continuation:
        source = Path(continuation["source_output"]).expanduser()
        source = (source if source.is_absolute() else ROOT / source).resolve()
        if source == output:
            raise ValueError("Continuation must use a new output directory")
        resume = load_resume_source(source, config, environment_config)
        if config["training"]["total_env_steps"] <= resume["payload"]["training_state"]["env_steps"]:
            raise ValueError("Continuation total budget must exceed the checkpoint's actual steps")
        budgets = curriculum.get("stage_step_budgets", [curriculum["max_stage_steps"]] * len(curriculum["stage_half_angles_degrees"]))
        if budgets[resume["stage_index"] - 1] <= resume["payload"]["training_state"]["stage_env_steps"]:
            raise ValueError("Continuation must extend the stopped stage's budget")
    # Validate the full orbit before writing artifacts or beginning training.
    for width in curriculum["stage_half_angles_degrees"]:
        for angle in evaluation_angles(width, stage_evaluation_settings(curriculum, width)[0]):
            arrival_position(environment_config, float(angle))
    set_seed(config["seed"], config["torch_num_threads"])
    output.mkdir(parents=True, exist_ok=True)
    for name, mapping in (("training_config.yaml", config), ("environment_config.yaml", environment_config)):
        (output / name).write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
    metadata = write_metadata(
        output / "metadata.json", config, environment_config, config_path=config_path,
        env_config_path=env_config_path, output_dir=output,
        extra={"training_mode": "arrival_direction_curriculum", "fresh_stage1": resume is None and warm is None,
               "initial_checkpoint": (str(warm["checkpoint"]) if warm is not None
                                      else None if resume is None else str(resume["checkpoint"])), "curriculum": curriculum,
               "resumed_from_output": None if resume is None else str(resume["source_output"]),
               "resume_checkpoint_sha256": None if resume is None else _sha(resume["checkpoint"]),
               "episode_randomization": "fixed at zero half-angle; otherwise continuous uniform azimuth; distance and height fixed",
               "evaluation_grid": "one fixed scene at zero; otherwise fixed endpoint-inclusive linspace; separate environment",
               "best_selection": "capture, lower attack failure, lower successful capture time, lower minimum distance, lower collision episode rate"},
    )
    if warm is not None:
        metadata.update(warm_start=dict(checkpoint=str(warm["checkpoint"]), checkpoint_sha256=warm["sha256"],
                        source_pretrained_env_steps=warm["source_env_steps"], source_update=warm["source_update"],
                        source_episodes=warm["source_episodes"], inherited_networks_and_both_adam=True,
                        inherited_global_rng=True, starts_new_episode=True,
                        discarded_source_partial_episode_steps=warm["discarded_partial_episode_steps"],
                        previous_stage_half_angle_degrees=config["warm_start"].get("previous_stage_half_angle_degrees")),
                        source_pretrained_env_steps=warm["source_env_steps"],
                        initial_policy_is_pretrained=True,
                        evaluation_timing=curriculum.get("evaluation_timing", "exact_steps"))
        write_json(output / "metadata.json", metadata)
    counters = dict(env_steps=0, update=0, episodes=0)
    results, parent, agent = [], None, None
    if warm is not None:
        parent = warm["checkpoint"]
    start_stage = 1
    if resume is not None:
        copy_resume_artifacts(resume, output)
        saved = resume["payload"]["training_state"]
        counters = {key: int(saved[key]) for key in counters}
        start_stage = int(resume["stage_index"])
        results = deepcopy(resume["summary"]["stages"][:-1])
        parent = None if not saved.get("parent_checkpoint") else Path(saved["parent_checkpoint"])
    overall_status = "running"
    with (output / "evaluation_history.csv").open("a" if resume is not None else "x", encoding="utf-8", newline="") as history_stream:
        history_writer = csv.DictWriter(history_stream, fieldnames=EVALUATION_FIELDS)
        if history_stream.tell() == 0:
            history_writer.writeheader()
        for stage_index, half_angle in enumerate(curriculum["stage_half_angles_degrees"]):
            stage_number = stage_index + 1
            if stage_number < start_stage:
                continue
            resuming_stage = resume is not None and stage_number == start_stage
            evaluation_count, promotion_threshold = stage_evaluation_settings(curriculum, half_angle)
            stage_budget = curriculum.get("stage_step_budgets", [curriculum["max_stage_steps"]] * len(curriculum["stage_half_angles_degrees"]))[stage_index]
            stage_path = output / "stage_{:02d}_pm{:g}".format(stage_number, half_angle)
            checkpoints = stage_path / "checkpoints"
            checkpoints.mkdir(parents=True, exist_ok=resuming_stage)
            source_state = None
            if resuming_stage:
                agent, payload = load_best_eval_checkpoint(resume["checkpoint"], config, config["device"])
                source_state = None if parent is None else torch.load(parent, map_location="cpu", weights_only=False)["training_state"]
            elif stage_index:
                agent, payload = load_best_eval_checkpoint(parent, config, config["device"])
                source_state = payload["training_state"]
            elif warm is not None:
                if _sha(warm["checkpoint"]) != warm["sha256"]:
                    raise ValueError("Warm-start checkpoint changed after preflight")
                agent, payload = load_best_eval_checkpoint(warm["checkpoint"], config, config["device"])
                source_state = payload["training_state"]
            stage_seed = config["seed"] + stage_index * curriculum["stage_sampling_seed_stride"]
            env = DirectionCurriculumEnv(env_config_path, half_angle, stage_seed)
            observations, info = env.reset(seed=stage_seed)
            if agent is None:
                observation_scale, state_scale = build_input_scales(env.config, config["normalization"])
                agent = MAPPOAgent(config, observation_dim=observations.shape[-1], state_dim=info["global_state"].shape[-1],
                                   action_dim=env.action_shape[-1], observation_scale=observation_scale,
                                   state_scale=state_scale, device=config["device"])
            agent.config = deepcopy(config)
            agent.actor.train()
            agent.critic.train()
            stage_steps = stage_updates = evaluations = 0
            recent = deque(maxlen=config["training"]["capture_rate_window"])
            episode = _episode_start(info)
            episode_angle = float(env.current_angle_degrees)
            episode_position = env.states[-1, :3].copy()
            buffer = RolloutBuffer(config["training"]["rollout_steps"], env.num_pursuers,
                                   observations.shape[-1], info["global_state"].shape[-1], env.action_shape[-1])
            gate = PromotionGate(promotion_threshold, curriculum["required_consecutive_evaluations"])
            best_summary = best_key = None
            best_path = checkpoints / "best_eval.pt"
            stage_status = "running"
            if resuming_stage:
                saved = payload["training_state"]
                observations, info = restore_environment(env, saved)
                stage_steps, stage_updates = int(saved["stage_env_steps"]), int(saved["stage_updates"])
                evaluations = int(resume["summary"]["stages"][-1]["evaluations"])
                recent = deque(deepcopy(saved["recent_episodes"]), maxlen=config["training"]["capture_rate_window"])
                episode = deepcopy(saved["partial_episode"])
                episode_angle = float(env.current_angle_degrees)
                episode_position = arrival_position(env.config, episode_angle)
                gate.consecutive = int(saved["consecutive_passes"])
                best_summary = deepcopy(resume["best_summary"])
                best_key = evaluation_key(best_summary)
                restore_rng_state(payload["rng_state"])

            def checkpoint_state(evaluation=None):
                state = dict(counters, stage_index=stage_number, stage_half_angle_degrees=float(half_angle),
                            stage_env_steps=stage_steps, stage_updates=stage_updates,
                            partial_episode=dict(episode), recent_episodes=list(recent),
                            current_episode_angle_degrees=float(env.current_angle_degrees),
                            environment_state=env.get_state(), environment_accelerations=env.accelerations.copy(),
                            environment_step_count=env.step_count, environment_rng_state=env.rng.bit_generator.state,
                            environment_internal=snapshot_environment(env),
                            angle_sampling_rng_state=deepcopy(env.angle_rng.bit_generator.state),
                            evaluation_summary=evaluation, best_eval_summary=best_summary,
                            best_selection="deterministic_current_stage_evaluation_lexicographic",
                            parent_checkpoint=None if parent is None else str(parent),
                            parent_checkpoint_sha256=None if parent is None else _sha(parent),
                            inherited_checkpoint_env_steps=None if source_state is None else source_state["env_steps"],
                            inherited_checkpoint_stage_steps=None if source_state is None else source_state.get("stage_env_steps"),
                            consecutive_passes=gate.consecutive)
                if warm is not None:
                    state.update(source_pretrained_env_steps=warm["source_env_steps"],
                                 lineage_env_steps=warm["source_env_steps"] + counters["env_steps"],
                                 warm_start_checkpoint=str(warm["checkpoint"]),
                                 warm_start_checkpoint_sha256=warm["sha256"],
                                 initial_policy_is_pretrained=True,
                                 discarded_source_partial_episode_steps=warm["discarded_partial_episode_steps"])
                return state

            agent.save(checkpoints / ("resume_initial.pt" if resuming_stage else "initial.pt"), checkpoint_state(), metadata)
            angles = evaluation_angles(half_angle, evaluation_count)
            write_json(stage_path / "evaluation_angles.json", angles.tolist())
            if warm is not None and stage_index == 0 and not resuming_stage:
                initial_path = stage_path / "evaluations" / "at_000000"
                initial_summary = evaluator(agent, env_config_path, half_angle, evaluation_count,
                                            curriculum["evaluation_seed"], initial_path / "current")
                if initial_summary["episodes"] != evaluation_count:
                    raise ValueError("Incomplete initial warm-start evaluation")
                if half_angle == 0 and initial_summary["capture_rate"] not in (0.0, 1.0):
                    raise ValueError("A single zero-angle episode must either capture or fail")
                best_key, best_summary = evaluation_key(initial_summary), deepcopy(initial_summary)
                initial_checkpoint = checkpoints / "evaluated_000000.pt"
                agent.save(initial_checkpoint, checkpoint_state(initial_summary), metadata)
                _copy_checkpoint(initial_checkpoint, best_path)
                (initial_path / "current").mkdir(parents=True, exist_ok=True)
                write_json(initial_path / "current" / "summary.json", initial_summary)
                write_json(initial_path / "current" / "checkpoint.json", dict(
                    checkpoint=str(initial_checkpoint), sha256=_sha(initial_checkpoint),
                    env_steps=0, stage_env_steps=0, initial_pretrained_baseline=True))
                history_writer.writerow(dict(stage_index=stage_number, half_angle_degrees=half_angle,
                    stage_env_steps=0, env_steps=0, update=0, role="initial", evaluated_half_angle_degrees=half_angle,
                    **{k: initial_summary[k] for k in ("capture_rate", "enemy_attack_success_rate", "average_capture_time", "average_min_distance", "collision_episode_rate")},
                    passed=None, consecutive_passes=0, checkpoint=str(initial_checkpoint)))
                previous_angle = config["warm_start"].get("previous_stage_half_angle_degrees")
                if previous_angle is not None and curriculum["monitor_previous_stage"]:
                    previous_count, _ = stage_evaluation_settings(curriculum, previous_angle)
                    initial_previous = evaluator(agent, env_config_path, previous_angle, previous_count,
                                                 curriculum["evaluation_seed"], initial_path / "previous")
                    if initial_previous["episodes"] != previous_count:
                        raise ValueError("Incomplete initial previous-stage evaluation")
                    evaluation_key(initial_previous)
                    (initial_path / "previous").mkdir(parents=True, exist_ok=True)
                    write_json(initial_path / "previous" / "summary.json", initial_previous)
                    write_json(initial_path / "previous" / "checkpoint.json", dict(
                        checkpoint=str(initial_checkpoint), sha256=_sha(initial_checkpoint),
                        initial_pretrained_baseline=True))
                    history_writer.writerow(dict(stage_index=stage_number, half_angle_degrees=half_angle,
                        stage_env_steps=0, env_steps=0, update=0, role="initial_previous", evaluated_half_angle_degrees=previous_angle,
                        **{k: initial_previous[k] for k in ("capture_rate", "enemy_attack_success_rate", "average_capture_time", "average_min_distance", "collision_episode_rate")},
                        passed=None, consecutive_passes=None, checkpoint=str(initial_checkpoint)))
                history_stream.flush()
            with (stage_path / "training_metrics.csv").open("a" if resuming_stage else "x", encoding="utf-8", newline="") as ep_stream, \
                    (stage_path / "updates.csv").open("a" if resuming_stage else "x", encoding="utf-8", newline="") as update_stream:
                ep_writer = csv.DictWriter(ep_stream, fieldnames=("stage_index", "half_angle_degrees", "stage_env_steps", "angle_degrees", "initial_attacker_x", "initial_attacker_y", "initial_attacker_z") + EPISODE_FIELDS)
                update_writer = csv.DictWriter(update_stream, fieldnames=("stage_index", "half_angle_degrees", "stage_env_steps", "stage_updates") + UPDATE_FIELDS)
                if ep_stream.tell() == 0:
                    ep_writer.writeheader()
                if update_stream.tell() == 0:
                    update_writer.writeheader()
                while stage_steps < stage_budget and counters["env_steps"] < config["training"]["total_env_steps"]:
                    previous_stage_steps = stage_steps
                    collect = rollout_collection_steps(curriculum, stage_steps, stage_budget,
                        config["training"]["total_env_steps"] - counters["env_steps"], buffer.capacity)
                    buffer.reset()
                    for _ in range(collect):
                        actions, log_probs, raw_actions, value = agent.act(observations, info["global_state"])
                        next_observations, rewards, terminated, truncated, next_info = env.step(actions)
                        next_value = 0.0 if terminated else agent.value(next_info["global_state"])
                        reward = _accumulate_episode(episode, rewards, next_info, env.num_pursuers)
                        buffer.add(observations, info["global_state"], raw_actions, log_probs,
                                   reward * config["normalization"]["reward_scale"], value, next_value, terminated, truncated)
                        counters["env_steps"] += 1
                        stage_steps += 1
                        observations, info = next_observations, next_info
                        if terminated or truncated:
                            counters["episodes"] += 1
                            recent.append(dict(captured=int(info["captured"]), enemy_attack_success=int(info["enemy_attack_success"]),
                                               truncated=int(truncated), episode_reward=episode["episode_reward"]))
                            ep_writer.writerow(dict(stage_index=stage_number, half_angle_degrees=half_angle,
                                                    stage_env_steps=stage_steps, angle_degrees=episode_angle,
                                                    initial_attacker_x=episode_position[0], initial_attacker_y=episode_position[1],
                                                    initial_attacker_z=episode_position[2],
                                                    **_episode_row(episode, info, truncated, recent, counters, env)))
                            ep_stream.flush()
                            observations, info = env.reset()
                            episode = _episode_start(info)
                            episode_angle = float(env.current_angle_degrees)
                            episode_position = env.states[-1, :3].copy()
                    buffer.compute_returns(config["ppo"]["gamma"], config["ppo"]["gae_lambda"])
                    losses = agent.update(buffer)
                    if not all(np.isfinite(float(v)) for v in losses.values()):
                        raise RuntimeError("Non-finite PPO update metric")
                    counters["update"] += 1
                    stage_updates += 1
                    capture_rate = float(np.mean([r["captured"] for r in recent])) if recent else None
                    enemy_rate = float(np.mean([r["enemy_attack_success"] for r in recent])) if recent else None
                    timeout_rate = float(np.mean([r["truncated"] for r in recent])) if recent else None
                    update_writer.writerow(dict(stage_index=stage_number, half_angle_degrees=half_angle,
                                                stage_env_steps=stage_steps, stage_updates=stage_updates, **counters,
                                                rollout_steps=buffer.size, capture_rate=capture_rate,
                                                mean_episode_reward=float(np.mean([r["episode_reward"] for r in recent])) if recent else None,
                                                interception_success_rate=capture_rate, enemy_attack_success_rate=enemy_rate,
                                                timeout_rate=timeout_rate, **losses))
                    update_stream.flush()
                    if stage_updates % config["training"]["checkpoint_interval_updates"] == 0:
                        agent.save(checkpoints / "update_{:06d}.pt".format(stage_updates), checkpoint_state(), metadata)
                    if stage_updates % config["training"]["log_interval_updates"] == 0:
                        print("stage={} +/-{:g} steps={}/{} actual_total_steps={} update={}".format(
                            stage_number, half_angle, stage_steps, stage_budget, counters["env_steps"], counters["update"]), flush=True)
                    if not evaluation_due(curriculum, previous_stage_steps, stage_steps, stage_budget,
                                          counters["env_steps"] >= config["training"]["total_env_steps"]):
                        continue
                    evaluation_path = stage_path / "evaluations" / "at_{:06d}".format(stage_steps)
                    summary = evaluator(agent, env_config_path, half_angle, evaluation_count,
                                        curriculum["evaluation_seed"], evaluation_path / "current")
                    if summary["episodes"] != evaluation_count:
                        raise ValueError("Incomplete current-stage evaluation")
                    key = evaluation_key(summary)
                    if half_angle == 0 and summary["capture_rate"] not in (0.0, 1.0):
                        raise ValueError("A single zero-angle episode must either capture or fail")
                    evaluations += 1
                    ready = gate.observe(summary["capture_rate"])
                    improved = best_key is None or key > best_key
                    if improved:
                        best_key, best_summary = key, deepcopy(summary)
                    evaluated_checkpoint = checkpoints / "evaluated_{:06d}.pt".format(stage_steps)
                    agent.save(evaluated_checkpoint, checkpoint_state(summary), metadata)
                    if improved:
                        _copy_checkpoint(evaluated_checkpoint, best_path)
                    for role, evaluated_angle, evaluation_summary in [("current", half_angle, summary)]:
                        directory = evaluation_path / role
                        directory.mkdir(parents=True, exist_ok=True)
                        write_json(directory / "summary.json", evaluation_summary)
                        write_json(directory / "checkpoint.json", {"checkpoint": str(evaluated_checkpoint), "sha256": _sha(evaluated_checkpoint), "env_steps": counters["env_steps"], "stage_env_steps": stage_steps})
                        history_writer.writerow(dict(stage_index=stage_number, half_angle_degrees=half_angle,
                                                     stage_env_steps=stage_steps, **{k: counters[k] for k in ("env_steps", "update")},
                                                     role=role, evaluated_half_angle_degrees=evaluated_angle,
                                                     **{k: evaluation_summary[k] for k in ("capture_rate", "enemy_attack_success_rate", "average_capture_time", "average_min_distance", "collision_episode_rate")},
                                                     passed=bool(evaluation_summary["capture_rate"] >= promotion_threshold),
                                                     consecutive_passes=gate.consecutive, checkpoint=str(evaluated_checkpoint)))
                    previous_angle = (curriculum["stage_half_angles_degrees"][stage_index - 1] if stage_index
                                      else config.get("warm_start", {}).get("previous_stage_half_angle_degrees"))
                    if previous_angle is not None and curriculum["monitor_previous_stage"]:
                        previous_count, _ = stage_evaluation_settings(curriculum, previous_angle)
                        monitored = evaluator(agent, env_config_path, previous_angle, previous_count,
                                              curriculum["evaluation_seed"], evaluation_path / "previous")
                        if monitored["episodes"] != previous_count:
                            raise ValueError("Incomplete previous-stage evaluation")
                        evaluation_key(monitored)
                        (evaluation_path / "previous").mkdir(parents=True, exist_ok=True)
                        write_json(evaluation_path / "previous" / "summary.json", monitored)
                        write_json(evaluation_path / "previous" / "checkpoint.json", {"checkpoint": str(evaluated_checkpoint), "sha256": _sha(evaluated_checkpoint)})
                        history_writer.writerow(dict(stage_index=stage_number, half_angle_degrees=half_angle,
                                                     stage_env_steps=stage_steps, **{k: counters[k] for k in ("env_steps", "update")},
                                                     role="previous", evaluated_half_angle_degrees=previous_angle,
                                                     **{k: monitored[k] for k in ("capture_rate", "enemy_attack_success_rate", "average_capture_time", "average_min_distance", "collision_episode_rate")},
                                                     passed=None, consecutive_passes=None, checkpoint=str(evaluated_checkpoint)))
                    history_stream.flush()
                    agent.save(checkpoints / "last.pt", checkpoint_state(summary), metadata)
                    print("evaluation stage={} steps={} capture={:.3f} streak={} best_capture={:.3f}".format(
                        stage_number, stage_steps, summary["capture_rate"], gate.consecutive, best_summary["capture_rate"]), flush=True)
                    if ready:
                        stage_status = "completed" if stage_index == len(curriculum["stage_half_angles_degrees"]) - 1 else "promoted"
                        break
                if stage_status == "running":
                    stage_status = "stopped_stage_budget" if stage_steps >= stage_budget else "stopped_total_budget"
            stage_result = dict(stage_index=stage_number, half_angle_degrees=float(half_angle), stage_env_steps=stage_steps,
                                stage_updates=stage_updates, evaluations=evaluations, consecutive_passes=gate.consecutive,
                                status=stage_status, selected_checkpoint=str(best_path), selected_checkpoint_sha256=_sha(best_path),
                                selected_evaluation=best_summary, parent_checkpoint=None if parent is None else str(parent),
                                parent_checkpoint_sha256=None if parent is None else _sha(parent),
                                inherited_checkpoint_env_steps=None if source_state is None else source_state["env_steps"],
                                discarded_partial_episode_steps=episode["episode_length"])
            if half_angle == 0:
                stage_result.update(evaluation_episodes=evaluation_count, promotion_capture_rate=promotion_threshold)
            if warm is not None:
                stage_result.update(source_pretrained_env_steps=warm["source_env_steps"],
                    evaluation_timing=curriculum.get("evaluation_timing", "exact_steps"),
                    previous_stage_half_angle_degrees=(curriculum["stage_half_angles_degrees"][stage_index - 1]
                        if stage_index else config["warm_start"].get("previous_stage_half_angle_degrees")),
                    initial_baseline_does_not_count_toward_promotion=True)
            results.append(stage_result)
            write_json(stage_path / "summary.json", stage_result)
            overall_status = (stage_status if stage_status in ("stopped_stage_budget", "stopped_total_budget", "completed")
                              else "stopped_total_budget" if counters["env_steps"] >= config["training"]["total_env_steps"] else "running")
            run_summary = dict(status=overall_status, **counters, stages=results, selected_checkpoint=str(best_path),
                               step_count_semantics="actual work consumed, not rolled back to selected lineage")
            if warm is not None:
                run_summary.update(source_pretrained_env_steps=warm["source_env_steps"],
                    lineage_env_steps=warm["source_env_steps"] + counters["env_steps"],
                    warm_start_checkpoint=str(warm["checkpoint"]),
                    warm_start_checkpoint_sha256=warm["sha256"])
            write_json(output / "curriculum_state.json", run_summary)
            if overall_status != "running":
                write_json(output / "summary.json", run_summary)
                run_post_training_analysis(output, config)
                return run_summary
            parent = best_path
    raise RuntimeError("Curriculum ended without a terminal status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--check-config", action="store_true", help="Validate and print the plan without creating outputs or training")
    args = parser.parse_args()
    if args.check_config:
        config, env_path = load_training_config(args.config)
        curriculum = validate_config(config)
        if "warm_start" in config:
            load_warm_start_source(config, yaml.safe_load(env_path.read_text(encoding="utf-8")))
        print(yaml.safe_dump(dict(environment_config=str(env_path), output_dir=config["output_dir"], curriculum=curriculum,
                                  total_training_cap=config["training"]["total_env_steps"], formal_training_started=False), sort_keys=False, allow_unicode=True))
    else:
        train_curriculum(args.config, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
