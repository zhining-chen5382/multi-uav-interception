"""Opt-in pretrained initialization and full-rollout curriculum scheduling.

Warm start transfers learned networks, Adam and global RNG state. It starts a
new task episode and new work counters; it is not same-episode continuation.
These helpers never alter the historical checkpoint or the physical task.
"""

import hashlib
from pathlib import Path
import re

import torch

from .direction_curriculum import _finite_state
from .experiment import ROOT


EVALUATION_TIMINGS = ("exact_steps", "after_full_rollout")


def rollout_collection_steps(curriculum, stage_steps, stage_budget,
                             remaining_total_steps, rollout_steps):
    """Keep full PPO batches unless a hard training budget ends the batch."""
    collect = min(rollout_steps, stage_budget - stage_steps, remaining_total_steps)
    if curriculum.get("evaluation_timing", "exact_steps") == "exact_steps":
        interval = curriculum["evaluation_interval_steps"]
        collect = min(collect, (stage_steps // interval + 1) * interval - stage_steps)
    return collect


def evaluation_due(curriculum, previous_stage_steps, stage_steps, stage_budget,
                   total_budget_exhausted=False):
    """Evaluate an updated policy once per crossed threshold, using actual steps.

The opt-in full-batch schedule also evaluates the true final hard-budget
checkpoint. ``interval >= rollout_steps`` is checked by the runner, so one
update cannot silently skip more than one nominal evaluation threshold.
"""
    interval = curriculum["evaluation_interval_steps"]
    if curriculum.get("evaluation_timing", "exact_steps") == "exact_steps":
        return stage_steps % interval == 0
    return (stage_steps // interval > previous_stage_steps // interval
            or stage_steps >= stage_budget or total_budget_exhausted)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_warm_start_source(config, environment_config):
    """Read and validate an external pretrained source without consuming RNG.

All physical and learning settings must match. Output location, training
budget, evaluation plan, and source-specific bookkeeping may differ. Actual
construction/loading is done only after this preflight by the existing loader.
"""
    settings = config.get("warm_start")
    _require(isinstance(settings, dict) and bool(settings), "warm_start must be a nonempty mapping")
    raw = settings.get("checkpoint")
    _require(isinstance(raw, str) and bool(raw.strip()), "warm_start.checkpoint must name a checkpoint")
    checkpoint = Path(raw).expanduser()
    checkpoint = (checkpoint if checkpoint.is_absolute() else ROOT / checkpoint).resolve()
    _require(checkpoint.is_file(), "Warm-start checkpoint does not exist: " + str(checkpoint))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = settings.get("expected_sha256")
    if expected is not None:
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                 "warm_start.expected_sha256 must be a lowercase SHA256")
        _require(digest == expected, "Warm-start checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict) and payload.get("format_version") == 1,
             "Unsupported warm-start checkpoint format")
    _finite_state(payload, "warm-start checkpoint")
    old_config = payload.get("config", {})
    for key in ("network", "ppo", "normalization", "seed", "device", "torch_num_threads"):
        _require(key in config and old_config.get(key) == config[key],
                 "Warm-start checkpoint learning setting differs: " + key)
    for key in ("rollout_steps", "log_interval_updates", "checkpoint_interval_updates", "capture_rate_window"):
        _require(old_config.get("training", {}).get(key) == config["training"].get(key),
                 "Warm-start checkpoint training setting differs: " + key)
    _require(payload.get("metadata", {}).get("environment_config") == environment_config,
             "Warm-start checkpoint environment_config differs from this task")
    _require(payload.get("dimensions") == dict(observation_dim=21, state_dim=24, action_dim=3),
             "Warm-start checkpoint dimensions differ from the baseline")
    for name in ("actor", "critic", "actor_optimizer", "critic_optimizer", "rng_state", "training_state"):
        _require(isinstance(payload.get(name), dict), "Warm-start checkpoint is missing " + name)
    for name in ("actor_optimizer", "critic_optimizer"):
        optimizer = payload[name]
        _require(bool(optimizer.get("state")) and bool(optimizer.get("param_groups")),
                 "Warm-start checkpoint is missing trained Adam state: " + name)
    _require(all(name in payload["rng_state"] for name in ("python", "numpy", "torch")),
             "Warm-start checkpoint is missing complete global RNG state")
    state = payload["training_state"]
    for key in ("env_steps", "update", "episodes"):
        value = state.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                 "Warm-start checkpoint has invalid " + key)
    _require(state["env_steps"] > 0 and state["update"] > 0, "Warm-start checkpoint must be a trained model")
    partial = state.get("partial_episode", {})
    discarded = partial.get("episode_length", state.get("environment_step_count", 0))
    _require(isinstance(discarded, int) and not isinstance(discarded, bool) and discarded >= 0,
             "Warm-start checkpoint has invalid partial episode length")
    return dict(checkpoint=checkpoint, payload=payload, sha256=digest,
                source_env_steps=state["env_steps"], source_update=state["update"],
                source_episodes=state["episodes"], discarded_partial_episode_steps=discarded)
