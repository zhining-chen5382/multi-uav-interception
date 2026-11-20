"""Paired fixed-zero PPO retention diagnostic; Adam state is the only treatment.

Evaluations at exact sample counts pause collection, never shorten a rollout.
At a full-batch boundary (and the hard budget), update first, then evaluate.
Intermediate evaluation checkpoints retain an unfinished buffer and are analysis
snapshots, not complete resume-training checkpoints.
"""

import argparse
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from algorithms.mappo import MAPPOAgent, RolloutBuffer
from envs import PursuitEnv
from scripts.train_mappo import EPISODE_FIELDS, UPDATE_FIELDS, _accumulate_episode, _episode_start
from scripts.verify_zero_direction_reproduction import first_difference
from utils.curriculum_warm_start import load_warm_start_source
from utils.direction_curriculum import restore_rng_state
from utils.evaluation import REWARD_METRIC_FIELDS, episode_outcome, run_episode, save_trajectory, summarize_episodes, write_json
from utils.experiment import ROOT, load_training_config, set_seed, write_metadata
from utils.post_training import run_post_training_analysis


DEFAULT_CONFIG = ROOT / "configs/mappo_policy_retention.yaml"
EVALUATION_FIELDS = (
    "group", "env_steps", "update", "last_update_env_steps", "pending_rollout_steps",
    "checked_after_update", "captured", "capture_time", "min_distance", "final_min_distance",
    "episode_reward", "episode_length", "enemy_attack_success", "truncated", "outcome",
    "collision_steps", "boundary_clip_count", "checkpoint", "trajectory",
)
DIAGNOSTIC_FIELDS = tuple(
    name + "_" + statistic
    for name in ("reward", "value", "return", "advantage")
    for statistic in ("mean", "std", "min", "max")
) + (
    "rollout_captures", "rollout_enemy_attacks", "actor_learning_rate", "critic_learning_rate",
    "actor_parameter_delta_l2", "critic_parameter_delta_l2", "actor_std_mean",
)
RETENTION_UPDATE_FIELDS = UPDATE_FIELDS + DIAGNOSTIC_FIELDS
FIXED_STD_UPDATE_FIELDS = (
    "group", "env_steps", "update", "multiplier",
    "log_std_x", "log_std_y", "log_std_z", "std_x", "std_y", "std_z",
    "log_std_requires_grad", "log_std_grad_is_none", "frozen_log_std_unchanged",
    "frozen_log_std_adam_unchanged", "frozen_log_std_adam_step",
    "actor_mean_parameter_delta_l2",
)


def _resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, name):
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             name + " must be a positive integer")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_state():
    return dict(python=random.getstate(), numpy=deepcopy(np.random.get_state()),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def validate_config(config, environment_config):
    """Validate all treatment settings and the original source before writing."""
    _require(isinstance(config.get("retention"), dict), "retention must be a mapping")
    for key in ("curriculum", "warm_start", "continuation", "initial_checkpoint", "resume_checkpoint"):
        _require(not config.get(key), "Retention cannot enable " + key)
    settings = config["retention"]
    groups = settings.get("groups")
    _require(isinstance(groups, list) and len(groups) == 2, "Retention requires the A/B pair")
    _require(groups == [dict(name="A", reset_adam=False), dict(name="B", reset_adam=True)],
             "Groups must be A inherited Adam and B reset Adam")
    _count(settings.get("evaluation_interval_steps"), "retention.evaluation_interval_steps")
    seed = settings.get("evaluation_seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
             "retention.evaluation_seed must be a nonnegative integer")
    _count(config["training"].get("total_env_steps"), "training.total_env_steps")
    for key in ("rollout_steps", "log_interval_updates", "checkpoint_interval_updates", "capture_rate_window"):
        _count(config["training"].get(key), "training." + key)
    _require(environment_config.get("attacker", {}).get("policy", {}).get("mode") == "direct",
             "Retention requires the unchanged fixed Direct Attacker task")
    source_config = deepcopy(config)
    source_config["warm_start"] = {name:settings[name] for name in ("checkpoint", "expected_sha256") if name in settings}
    return load_warm_start_source(source_config, environment_config)


def initialize_agent(config, source, reset_adam, *, actor_learning_rate=None,
                     frozen_std_multiplier=None):
    """Fresh-load each treatment, retaining all Adam param-group settings.

    Optimizer tensors must not alias a shared in-memory source payload: Adam
    mutates its moments in place. Each group consequently reads the file anew.
    """
    _require(isinstance(reset_adam, bool), "reset_adam must be boolean")
    _require(_sha(source["checkpoint"]) == source["sha256"], "Source checkpoint changed after validation")
    agent, payload = MAPPOAgent.from_checkpoint(source["checkpoint"], device=config["device"])
    if reset_adam:
        agent.actor_optimizer.state.clear()
        agent.critic_optimizer.state.clear()
    if actor_learning_rate is not None:
        _require(not reset_adam, "Actor-LR treatment must retain the original Adam state")
        _require(not isinstance(actor_learning_rate, bool)
                 and np.isfinite(actor_learning_rate) and actor_learning_rate > 0,
                 "actor_learning_rate must be finite and positive")
        for parameters in agent.actor_optimizer.param_groups:
            parameters["lr"] = float(actor_learning_rate)
    if frozen_std_multiplier is not None:
        _require(not reset_adam, "Fixed-std treatment must retain the original Adam state")
        _require(not isinstance(frozen_std_multiplier, bool)
                 and isinstance(frozen_std_multiplier, (int, float))
                 and np.isfinite(frozen_std_multiplier) and frozen_std_multiplier > 0,
                 "frozen_std_multiplier must be finite and positive")
        with torch.no_grad():
            shifted = agent.actor.log_std + math.log(float(frozen_std_multiplier))
            _require(bool(torch.isfinite(shifted).all())
                     and bool((shifted >= agent.actor.min_log_std).all())
                     and bool((shifted <= agent.actor.max_log_std).all()),
                     "Requested std shift exceeds the actor's existing log_std clamp")
            agent.actor.log_std.copy_(shifted)
        # Keep the original optimizer parameter list and moments/counters.
        # Adam skips parameters whose gradient is None, including weight decay.
        agent.actor.log_std.grad = None
        agent.actor.log_std.requires_grad_(False)
    agent.config = deepcopy(config)
    agent.ppo = agent.config["ppo"]
    agent.actor.train()
    agent.critic.train()
    restore_rng_state(payload["rng_state"])
    return agent, payload


def evaluate_once(agent, env_config_path, seed, output_dir):
    """A full fixed-zero deterministic episode without disturbing training RNG."""
    output = Path(output_dir)
    _require(not output.exists(), "Refusing to overwrite evaluation: " + str(output))
    output.mkdir(parents=True)
    before = _rng_state()
    modes = (agent.actor.training, agent.critic.training)
    try:
        env = PursuitEnv(env_config_path)
        agent.actor.eval()
        agent.critic.eval()
        with torch.inference_mode():
            metrics, trajectory = run_episode(env, agent, seed)
        save_trajectory(output / "first_episode.npz", trajectory, metrics, seed)
        summary = summarize_episodes([metrics])
        write_json(output / "summary.json", summary)
        write_json(output / "episode_metrics.json", metrics)
        with (output / "evaluation_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("episode", "seed") + tuple(metrics))
            writer.writeheader()
            writer.writerow(dict(episode=1, seed=seed, **metrics))
        return metrics, summary
    finally:
        restore_rng_state(before)
        agent.actor.train(modes[0])
        agent.critic.train(modes[1])


def _optimizer_description(optimizer):
    state = optimizer.state_dict()
    steps = [float(entry["step"].item() if isinstance(entry["step"], torch.Tensor) else entry["step"])
             for entry in state["state"].values() if "step" in entry]
    return dict(parameter_states=len(state["state"]), step_values=sorted(set(steps)),
                param_groups=deepcopy(state["param_groups"]))


def _delta(network, reference):
    with torch.no_grad():
        return float(sum((parameter.detach().cpu().double() - reference[name].cpu().double()).square().sum().item()
                         for name, parameter in network.named_parameters()) ** 0.5)


def _fixed_std_reference(agent):
    parameter = agent.actor.log_std
    _require(any(candidate is parameter for group in agent.actor_optimizer.param_groups
                 for candidate in group["params"]), "Fixed log_std is missing from the original Adam parameter list")
    return dict(log_std=parameter.detach().cpu().clone(),
                adam_state=deepcopy(agent.actor_optimizer.state.get(parameter, {})))


def _verify_fixed_std(agent, reference):
    parameter = agent.actor.log_std
    checks = dict(log_std_requires_grad=parameter.requires_grad,
                  log_std_grad_is_none=parameter.grad is None,
                  frozen_log_std_unchanged=torch.equal(parameter.detach().cpu(), reference["log_std"]),
                  frozen_log_std_adam_unchanged=first_difference(
                      agent.actor_optimizer.state.get(parameter, {}), reference["adam_state"], "log_std_adam") is None)
    _require(not checks["log_std_requires_grad"] and all(value for name, value in checks.items()
                                                        if name != "log_std_requires_grad"),
             "Fixed log_std or its retained Adam state changed: " + repr(checks))
    _require(any(candidate is parameter for group in agent.actor_optimizer.param_groups
                 for candidate in group["params"]), "Fixed log_std was removed from Adam")
    return checks


def _fixed_std_row(agent, reference, source_actor, multiplier, counters, group):
    checks = _verify_fixed_std(agent, reference)
    logs = agent.actor.log_std.detach().cpu()
    stds = logs.clamp(agent.actor.min_log_std, agent.actor.max_log_std).exp()
    step = reference["adam_state"].get("step", 0)
    step = float(step.item() if isinstance(step, torch.Tensor) else step)
    with torch.no_grad():
        mean_delta = float(sum((parameter.detach().cpu().double() - source_actor[name].cpu().double()).square().sum().item()
                              for name, parameter in agent.actor.named_parameters() if name != "log_std") ** 0.5)
    return dict(group=group, env_steps=counters["env_steps"], update=counters["update"],
                multiplier=float(multiplier), **checks, frozen_log_std_adam_step=step,
                actor_mean_parameter_delta_l2=mean_delta,
                **{"log_std_" + axis:float(logs[i]) for i, axis in enumerate("xyz")},
                **{"std_" + axis:float(stds[i]) for i, axis in enumerate("xyz")})


def _buffer_diagnostics(buffer, captures, attacks):
    result = {}
    for name, key in (("reward", "rewards"), ("value", "values"), ("return", "returns"), ("advantage", "advantages")):
        values = np.asarray(getattr(buffer, key)[:buffer.size], dtype=np.float64)
        result.update({name + "_mean":float(values.mean()), name + "_std":float(values.std()),
                       name + "_min":float(values.min()), name + "_max":float(values.max())})
    result.update(rollout_captures=captures, rollout_enemy_attacks=attacks)
    return result


def _save_first_rollout(output, buffer, actions, next_states):
    names = ("observations", "states", "raw_actions", "log_probs", "rewards", "values",
             "next_values", "advantages", "returns", "terminated", "truncated")
    arrays = {name:getattr(buffer, name)[:buffer.size].copy() for name in names}
    arrays.update(actions=np.asarray(actions), next_states=np.asarray(next_states))
    np.savez_compressed(output / "first_rollout.npz", **arrays)
    write_json(output / "first_rollout_manifest.json", dict(rollout_steps=buffer.size,
        array_sha256={name:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for name,value in arrays.items()},
        purpose="Compare the identical pre-first-update stochastic training data across A and B; no group or time metadata in arrays."))


def _episode_row(episode, info, truncated, counters, recent, env):
    length = episode["episode_length"]
    captured = bool(info["captured"])
    rate = float(np.mean([entry["captured"] for entry in recent]))
    enemy_rate = float(np.mean([entry["enemy_attack_success"] for entry in recent]))
    timeout = float(np.mean([entry["truncated"] for entry in recent]))
    return dict(episode=counters["episodes"], env_steps=counters["env_steps"],
        episode_reward=episode["episode_reward"], episode_length=length,
        captured=int(captured), truncated=int(truncated), capture_rate=rate,
        capture_time=length * env.dt if captured else None,
        min_distance=episode["min_distance"], final_min_distance=info["min_distance"],
        collision_steps=episode["collision_steps"], collision_step_rate=episode["collision_steps"] / length,
        average_pairwise_pursuer_distance=episode["pairwise_distance_sum"] / length,
        interception_success=int(captured), enemy_attack_success=int(info["enemy_attack_success"]),
        attacker_reached_base=int(info["attacker_reached_base"]), interception_failure=int(info["interception_failure"]),
        outcome=episode_outcome(info, truncated), distance_to_base=info["distance_to_base"],
        interception_time=length * env.dt if captured else None,
        interception_success_rate=rate, enemy_attack_success_rate=enemy_rate, timeout_rate=timeout,
        **{key:episode[key] for key in REWARD_METRIC_FIELDS})


def _evaluation_key(metrics):
    time = metrics["capture_time"] if metrics["captured"] else 0.0
    return (int(metrics["captured"]), -int(metrics["enemy_attack_success"]), -time,
            -metrics["min_distance"], -metrics["collision_steps"])


def run_group(config, env_config_path, source, group, output_dir, evaluation_fn=None):
    """Train one treatment; source weights remain immutable throughout."""
    output = Path(output_dir).resolve()
    _require(not output.exists(), "Use a new retention group output: " + str(output))
    group_config = deepcopy(config)
    group_config["output_dir"] = str(output)
    group_config["retention_group"] = deepcopy(group)
    actor_learning_rate = group.get("actor_learning_rate")
    frozen_std_multiplier = group.get("frozen_std_multiplier")
    if actor_learning_rate is not None:
        group_config["ppo"]["actor_learning_rate"] = float(actor_learning_rate)
    total = config["training"]["total_env_steps"]
    interval = config["retention"]["evaluation_interval_steps"]
    evaluation_seed = config["retention"]["evaluation_seed"]
    agent, inherited = initialize_agent(group_config, source, group["reset_adam"],
                                        actor_learning_rate=actor_learning_rate,
                                        frozen_std_multiplier=frozen_std_multiplier)
    fixed_std_reference = (_fixed_std_reference(agent) if frozen_std_multiplier is not None else None)
    frozen_std = None
    if fixed_std_reference is not None:
        frozen_std = dict(multiplier=float(frozen_std_multiplier),
                         source_log_std=inherited["actor"]["log_std"].cpu().tolist(),
                         applied_log_std=agent.actor.log_std.detach().cpu().tolist(),
                         applied_std=agent.actor.log_std.detach().cpu().exp().tolist(),
                         requires_grad=False, optimizer_parameter_and_state_retained=True)
    env = PursuitEnv(env_config_path)
    observations, info = env.reset(seed=config["seed"])
    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    for name, mapping in (("training_config.yaml", group_config), ("environment_config.yaml", env.config)):
        (output / name).write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
    metadata = write_metadata(output / "metadata.json", group_config, env.config,
        config_path=group_config.get("retention_config_source", DEFAULT_CONFIG),
        env_config_path=env_config_path, output_dir=output,
        extra=dict(training_mode=("fixed_zero_frozen_std_retention" if frozen_std is not None else
                                  "fixed_zero_actor_lr_retention" if actor_learning_rate is not None
                                  else "fixed_zero_policy_retention"), retention_group=group,
            source_checkpoint=str(source["checkpoint"]), source_checkpoint_sha256=source["sha256"],
            source_pretrained_env_steps=source["source_env_steps"], starts_new_episode=True,
            discarded_source_partial_episode_steps=source["discarded_partial_episode_steps"],
            evaluation_timing="Exact sample milestones pause collection without splitting rollouts; batch/hard-budget endpoints are evaluated after update.",
            comparison=("Only initial std multiplier varies; log_std is fixed in all groups, with both original Adam states retained."
                        if frozen_std is not None else
                        "Only Actor learning rate varies; both original Adam states and all other param-group settings are inherited."
                        if actor_learning_rate is not None else
                        "Only original Adam moments and step counters are reset in B; all param-group settings are inherited.")))
    expected_actor = inherited["actor"]
    if frozen_std is not None:
        expected_actor = deepcopy(expected_actor)
        expected_actor["log_std"] = inherited["actor"]["log_std"] + math.log(float(frozen_std_multiplier))
    proofs = {name:first_difference(getattr(agent,name).state_dict(), expected_actor if name == "actor" else inherited[name], name)
              for name in ("actor", "critic")}
    proofs["global_rng"] = first_difference(_rng_state(), inherited["rng_state"], "rng_state")
    for name in ("actor_optimizer", "critic_optimizer"):
        actual = getattr(agent,name).state_dict()
        expected_groups = deepcopy(inherited[name]["param_groups"])
        if name == "actor_optimizer" and actor_learning_rate is not None:
            for parameters in expected_groups:
                parameters["lr"] = float(actor_learning_rate)
        proofs[name + "_param_groups"] = first_difference(actual["param_groups"], expected_groups, name)
        proofs[name + "_state"] = (None if not actual["state"] else "Expected empty Adam state") if group["reset_adam"] else first_difference(actual["state"], inherited[name]["state"], name)
    _require(all(value is None for value in proofs.values()), "Initial treatment isolation failed: " + repr(proofs))
    initialization_proof = dict(all_checks_passed=True,
        group=group["name"], reset_adam=group["reset_adam"], checks=proofs,
        actor_optimizer=_optimizer_description(agent.actor_optimizer), critic_optimizer=_optimizer_description(agent.critic_optimizer),
        source_pretrained_env_steps=source["source_env_steps"], new_work_steps=0,
        original_source_file_sha256=source["sha256"], starts_new_episode=True,
        discarded_source_partial_episode_steps=source["discarded_partial_episode_steps"])
    if frozen_std is not None:
        initialization_proof["frozen_std"] = frozen_std
        initialization_proof["fixed_std_checks"] = _verify_fixed_std(agent, fixed_std_reference)
        metadata["frozen_std"] = deepcopy(frozen_std)
        write_json(output / "metadata.json", metadata)
    write_json(output / "initialization_verification.json", initialization_proof)
    buffer = RolloutBuffer(config["training"]["rollout_steps"], env.num_pursuers,
                          observations.shape[-1], info["global_state"].shape[-1], env.action_shape[-1])
    recent = deque(maxlen=config["training"]["capture_rate_window"])
    counters = dict(env_steps=0, update=0, episodes=0)
    episode = _episode_start(info)
    last_update_steps = 0
    best_key = best_summary = None
    best_path = checkpoint_dir / "best.pt"
    evaluator = evaluate_once if evaluation_fn is None else evaluation_fn
    evaluation_rows = []

    def checkpoint_state(evaluation=None):
        state = dict(counters, source_pretrained_env_steps=source["source_env_steps"],
            retention_group=group["name"], reset_adam=group["reset_adam"],
            partial_episode=deepcopy(episode), recent_episodes=list(recent),
            environment_state=env.get_state(), environment_accelerations=env.accelerations.copy(),
            environment_step_count=env.step_count, environment_rng_state=deepcopy(env.rng.bit_generator.state),
            last_update_env_steps=last_update_steps,
            pending_rollout_steps=counters["env_steps"] - last_update_steps,
            fully_resumable=False, checkpoint_purpose="Inference/retention snapshot; pending on-policy buffer is not serialized.",
            evaluation_summary=evaluation, best_eval_summary=best_summary,
            best_selection="deterministic_capture_then_time_distance_collision")
        if frozen_std is not None:
            state["frozen_std"] = deepcopy(frozen_std)
            state["frozen_std_checks"] = _verify_fixed_std(agent, fixed_std_reference)
            state["fixed_log_std_training_requires_reapplication_after_plain_checkpoint_load"] = True
        return state

    agent.save(checkpoint_dir / "initial.pt", checkpoint_state(), metadata)
    with (output / "evaluations.csv").open("x", encoding="utf-8", newline="") as eval_stream, \
            (output / "training_metrics.csv").open("x", encoding="utf-8", newline="") as episode_stream, \
            (output / "updates.csv").open("x", encoding="utf-8", newline="") as update_stream, \
            ((output / "fixed_std_updates.csv").open("x", encoding="utf-8", newline="")
             if frozen_std is not None else nullcontext()) as fixed_std_stream:
        eval_writer = csv.DictWriter(eval_stream, fieldnames=EVALUATION_FIELDS)
        ep_writer = csv.DictWriter(episode_stream, fieldnames=EPISODE_FIELDS)
        up_writer = csv.DictWriter(update_stream, fieldnames=RETENTION_UPDATE_FIELDS)
        for writer in (eval_writer, ep_writer, up_writer):
            writer.writeheader()
        fixed_std_writer = None
        if fixed_std_stream is not None:
            fixed_std_writer = csv.DictWriter(fixed_std_stream, fieldnames=FIXED_STD_UPDATE_FIELDS)
            fixed_std_writer.writeheader()

        def evaluate_current():
            nonlocal best_key, best_summary
            step = counters["env_steps"]
            directory = output / "evaluations" / "at_{:06d}".format(step)
            before_rng = _rng_state()
            modes = (agent.actor.training, agent.critic.training)
            try:
                metrics, summary = evaluator(agent, env_config_path, evaluation_seed, directory)
            finally:
                # This outer guard also isolates deliberately injected test evaluators.
                restore_rng_state(before_rng)
                agent.actor.train(modes[0])
                agent.critic.train(modes[1])
            key = _evaluation_key(metrics)
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_summary = key, deepcopy(summary)
            checkpoint = checkpoint_dir / "evaluated_{:06d}.pt".format(step)
            agent.save(checkpoint, checkpoint_state(summary), metadata)
            if improved:
                temporary = best_path.with_name("best.pt.tmp")
                temporary.write_bytes(checkpoint.read_bytes())
                temporary.replace(best_path)
            row = dict(group=group["name"], **{name:counters[name] for name in ("env_steps", "update")},
                last_update_env_steps=last_update_steps, pending_rollout_steps=step-last_update_steps,
                checked_after_update=(step == last_update_steps),
                **{name:metrics[name] for name in EVALUATION_FIELDS[6:17]},
                checkpoint=str(checkpoint), trajectory=str(directory / "first_episode.npz"))
            eval_writer.writerow(row)
            eval_stream.flush()
            evaluation_rows.append(row)
            print("group={} eval_added_steps={} last_update_steps={} pending={} captured={} min_d={:.6f}".format(
                group["name"], step, last_update_steps, step-last_update_steps, metrics["captured"], metrics["min_distance"]), flush=True)

        evaluate_current()
        while counters["env_steps"] < total:
            collect = min(buffer.capacity, total - counters["env_steps"])
            buffer.reset()
            captures = attacks = 0
            first_actions, first_next_states = [], []
            for _ in range(collect):
                state = info["global_state"]
                actions, log_probs, raw_actions, value = agent.act(observations, state)
                next_observations, rewards, terminated, truncated, next_info = env.step(actions)
                next_value = 0.0 if terminated else agent.value(next_info["global_state"])
                reward = _accumulate_episode(episode, rewards, next_info, env.num_pursuers)
                buffer.add(observations, state, raw_actions, log_probs,
                    reward * config["normalization"]["reward_scale"], value, next_value, terminated, truncated)
                if counters["update"] == 0:
                    first_actions.append(actions.copy())
                    first_next_states.append(next_info["global_state"].copy())
                counters["env_steps"] += 1
                observations, info = next_observations, next_info
                if terminated or truncated:
                    counters["episodes"] += 1
                    captures += int(info["captured"])
                    attacks += int(info["enemy_attack_success"])
                    recent.append(dict(captured=int(info["captured"]), enemy_attack_success=int(info["enemy_attack_success"]),
                        truncated=int(truncated), episode_reward=episode["episode_reward"]))
                    ep_writer.writerow(_episode_row(episode, info, truncated, counters, recent, env))
                    episode_stream.flush()
                    observations, info = env.reset()
                    episode = _episode_start(info)
                if counters["env_steps"] % interval == 0 and buffer.size < collect:
                    evaluate_current()
            buffer.compute_returns(config["ppo"]["gamma"], config["ppo"]["gae_lambda"])
            diagnostics = _buffer_diagnostics(buffer, captures, attacks)
            if counters["update"] == 0:
                _save_first_rollout(output, buffer, first_actions, first_next_states)
                if frozen_std is not None:
                    manifest_path = output / "first_rollout_manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["purpose"] = ("Actual first on-policy rollout at this group's fixed std; "
                                           "different std treatments deliberately produce different actions and states.")
                    write_json(manifest_path, manifest)
            if fixed_std_reference is not None:
                _verify_fixed_std(agent, fixed_std_reference)
            losses = agent.update(buffer)
            _require(all(np.isfinite(float(value)) for value in losses.values()), "Non-finite PPO update")
            counters["update"] += 1
            last_update_steps = counters["env_steps"]
            if fixed_std_writer is not None:
                fixed_std_writer.writerow(_fixed_std_row(agent, fixed_std_reference, inherited["actor"],
                                                        frozen_std_multiplier, counters, group["name"]))
                fixed_std_stream.flush()
            diagnostics.update(actor_learning_rate=agent.actor_optimizer.param_groups[0]["lr"],
                critic_learning_rate=agent.critic_optimizer.param_groups[0]["lr"],
                actor_parameter_delta_l2=_delta(agent.actor, inherited["actor"]),
                critic_parameter_delta_l2=_delta(agent.critic, inherited["critic"]),
                actor_std_mean=float(agent.actor.log_std.detach().clamp(
                    config["network"]["min_log_std"], config["network"]["max_log_std"]).exp().mean().item()))
            rate = float(np.mean([entry["captured"] for entry in recent])) if recent else None
            enemy = float(np.mean([entry["enemy_attack_success"] for entry in recent])) if recent else None
            timeout = float(np.mean([entry["truncated"] for entry in recent])) if recent else None
            row = dict(counters, rollout_steps=buffer.size, capture_rate=rate,
                mean_episode_reward=float(np.mean([entry["episode_reward"] for entry in recent])) if recent else None,
                interception_success_rate=rate, enemy_attack_success_rate=enemy, timeout_rate=timeout,
                **losses, **diagnostics)
            up_writer.writerow(row)
            update_stream.flush()
            if counters["update"] % config["training"]["checkpoint_interval_updates"] == 0:
                agent.save(checkpoint_dir / "update_{:06d}.pt".format(counters["update"]), checkpoint_state(), metadata)
            if counters["env_steps"] % interval == 0 or counters["env_steps"] == total:
                evaluate_current()
            if counters["update"] % config["training"]["log_interval_updates"] == 0:
                print("group={} update={} added_steps={}/{} actor_loss={:.6f} value_loss={:.6f} entropy={:.6f}".format(
                    group["name"], counters["update"], counters["env_steps"], total,
                    losses["actor_loss"], losses["value_loss"], losses["entropy"]), flush=True)
    final_summary = deepcopy(best_summary)
    final_evaluation = output / "evaluations" / "at_{:06d}".format(total) / "summary.json"
    if final_evaluation.is_file():
        final_summary = json.loads(final_evaluation.read_text(encoding="utf-8"))
    agent.save(checkpoint_dir / "final.pt", checkpoint_state(final_summary), metadata)
    _require(_sha(source["checkpoint"]) == source["sha256"], "Original checkpoint changed during retention training")
    summary = dict(status="completed", group=group["name"], reset_adam=group["reset_adam"], **counters,
        source_pretrained_env_steps=source["source_env_steps"], source_checkpoint=str(source["checkpoint"]),
        source_checkpoint_sha256=source["sha256"], best_checkpoint=str(best_path),
        final_checkpoint=str(checkpoint_dir / "final.pt"), evaluations=len(evaluation_rows),
        initial_capture=bool(evaluation_rows[0]["captured"]), final_capture=bool(evaluation_rows[-1]["captured"]),
        first_failed_evaluation_steps=next((row["env_steps"] for row in evaluation_rows if not row["captured"]), None),
        evaluation_semantics="Exact sample milestones, model after last completed PPO update; no evaluation-driven batch truncation.",
        incomplete_buffer_checkpoints_are_not_resume_checkpoints=True)
    if frozen_std is not None:
        summary["frozen_std"] = deepcopy(frozen_std)
        summary["frozen_std_verification"] = dict(all_checks_passed=True,
            checks=_verify_fixed_std(agent, fixed_std_reference), checked_after_every_update=True,
            mean_network_and_critic_remain_trainable=True,
            ordinary_actor_parameter_delta_includes_initial_log_std_shift=True,
            mean_only_parameter_delta_csv="fixed_std_updates.csv")
    write_json(output / "summary.json", summary)
    run_post_training_analysis(output, group_config)
    return summary


def run_experiment(config_path=DEFAULT_CONFIG, *, output_dir=None, seed=None):
    config_path = _resolve(config_path)
    config, env_path = load_training_config(config_path)
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if seed is not None:
        config["seed"] = seed
    environment = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    source = validate_config(config, environment)
    output = _resolve(config["output_dir"])
    _require(not output.exists(), "Use a new retention experiment output: " + str(output))
    _require(output != source["checkpoint"].parent.parent
             and source["checkpoint"].parent.parent not in output.parents,
             "Retention output must be independent of the source run")
    config["retention_config_source"] = str(config_path)
    set_seed(config["seed"], config["torch_num_threads"])
    output.mkdir(parents=True)
    write_metadata(output / "metadata.json", config, environment,
        config_path=config_path, env_config_path=env_path, output_dir=output,
        kind="policy_retention_comparison", extra=dict(source_checkpoint=str(source["checkpoint"]),
            source_checkpoint_sha256=source["sha256"], source_pretrained_env_steps=source["source_env_steps"],
            treatment="A original Adam; B zero Adam state; identical inherited weights/global RNG/param-group settings"))
    results = [run_group(config, env_path, source, group, output / group["name"])
               for group in config["retention"]["groups"]]
    with np.load(output / "A/first_rollout.npz", allow_pickle=False) as left, \
            np.load(output / "B/first_rollout.npz", allow_pickle=False) as right:
        equal = {name:np.array_equal(left[name], right[name]) for name in left.files}
    _require(all(equal.values()), "A/B first stochastic rollout differed before the optimizer treatment")
    summary = dict(status="completed", output_dir=str(output), groups=results,
        first_rollout_array_equality=equal, source_checkpoint_sha256=source["sha256"],
        source_unchanged=_sha(source["checkpoint"]) == source["sha256"],
        curriculum_training_started=False, learning_rates_changed=False)
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
            source_checkpoint=str(source["checkpoint"]), source_sha256=source["sha256"], groups=config["retention"]["groups"],
            evaluation_interval_steps=config["retention"]["evaluation_interval_steps"], formal_training_started=False), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(run_experiment(args.config, output_dir=args.output_dir, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
