"""Lossless continuation helpers for an existing direction-curriculum run.

Reading a source run never constructs a policy or changes global random state.
The legacy fallback is intentionally restricted to provable absolute-distance,
Direct-Attacker episodes; it refuses to guess missing boundary statistics.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from envs.attacker_policy import DirectAttackerPolicy
from envs.dynamics import integrate
from .direction_curriculum import DirectionCurriculumEnv, arrival_position, _finite_state


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mapping(value, name):
    _require(isinstance(value, dict), name + " must be a mapping")
    return value


def _number(value, name):
    _require(not isinstance(value, (bool, np.bool_)) and not np.iscomplexobj(value), name + " must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " must be finite") from error
    _require(np.isfinite(result), name + " must be finite")
    return result


def _count(value, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and value >= 0,
             name + " must be a nonnegative integer")
    return int(value)


def _array(value, shape, name):
    _require(not np.iscomplexobj(value), name + " must be real and finite")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " must be finite with shape " + str(shape)) from error
    _require(result.shape == shape and bool(np.isfinite(result).all()), name + " must be finite with shape " + str(shape))
    return result.copy()


def _json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    _mapping(result, str(path))
    _finite_state(result, str(path))
    return result


def _yaml(path):
    return _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), str(path))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generator(state, name):
    _mapping(state, name)
    kind = state.get("bit_generator")
    _require(kind in ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"), name + " has unsupported bit generator")
    try:
        generator = np.random.Generator(getattr(np.random, kind)())
        generator.bit_generator.state = deepcopy(state)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(name + " is invalid") from error
    return generator


def snapshot_environment(env):
    """Capture all simulator fields required to continue the same episode."""
    _require(env._has_reset, "Cannot snapshot an environment before reset")
    snapshot = dict(
        format_version=1, states=env.states.copy(), accelerations=env.accelerations.copy(),
        step_count=int(env.step_count), current_angle_degrees=float(env.current_angle_degrees),
        _initial_states=env._initial_states.copy(),
        _boundary_clip_counts=env._boundary_clip_counts.copy(), _boundary_clipped=env._boundary_clipped.copy(),
        _episode_reward_components=deepcopy(env._episode_reward_components),
        _has_reset=bool(env._has_reset), _done=bool(env._done),
        rng_state=deepcopy(env.rng.bit_generator.state), angle_rng_state=deepcopy(env.angle_rng.bit_generator.state),
    )
    _finite_state(snapshot, "environment_internal")
    return snapshot


def _legacy_internal(env, state, states, accelerations, steps, angle, initial):
    _require(env.absolute_distance_enabled and type(env.attacker_policy) is DirectAttackerPolicy,
             "Legacy resume requires absolute-distance reward and unchanged Direct Attacker")
    partial = _mapping(state.get("partial_episode"), "partial_episode")
    _require(_count(partial.get("episode_length"), "partial_episode.episode_length") == steps,
             "partial_episode.episode_length disagrees with environment_step_count")
    _require(steps < env.max_episode_steps, "Legacy partial episode must be unfinished")
    terminal = _number(partial.get("terminal_reward"), "partial_episode.terminal_reward")
    _require(terminal == 0, "Legacy partial episode contains terminal reward")
    totals = dict(
        absolute_distance_cost=_number(partial.get("cumulative_absolute_distance_cost"), "cumulative_absolute_distance_cost"),
        survival=_number(partial.get("cumulative_survival_penalty"), "cumulative_survival_penalty"),
        collision=_number(partial.get("cumulative_collision_penalty"), "cumulative_collision_penalty"),
        capture=0.0, attack_failure=0.0,
    )
    _require(set(totals) == set(env._reward_terms), "Legacy reward schema is unsupported")
    _require(np.isclose(sum(totals.values()), _number(partial.get("episode_reward"), "partial_episode.episode_reward"), rtol=1e-12, atol=1e-8),
             "Legacy partial weighted reward components do not sum to episode_reward")
    collision_steps = _count(partial.get("collision_steps"), "partial_episode.collision_steps")
    _require(collision_steps <= steps and np.isclose(totals["collision"], collision_steps * env.reward_coefficients["collision_penalty"], rtol=1e-12, atol=1e-8),
             "Legacy collision reward is inconsistent")
    _require(np.isclose(totals["survival"], steps * env.dt * env.reward_coefficients["survival_penalty_per_second"], rtol=1e-12, atol=1e-8),
             "Legacy survival reward is inconsistent")
    _require(totals["absolute_distance_cost"] <= 0, "Legacy distance cost must be nonpositive")
    # Every possible constrained pursuer action remains within these envelopes.
    # This proves absence of clipping without needing historical actor actions.
    limits = env.pursuer_limits
    elapsed = steps * env.dt
    lower_displacement = np.array([-limits.max_horizontal_speed, -limits.max_horizontal_speed, -limits.max_descent_speed]) * elapsed
    upper_displacement = np.array([limits.max_horizontal_speed, limits.max_horizontal_speed, limits.max_climb_speed]) * elapsed
    _require(np.all(initial[:-1, :3] + lower_displacement >= env.bounds_low)
             and np.all(initial[:-1, :3] + upper_displacement <= env.bounds_high),
             "Cannot prove zero legacy pursuer boundary clipping from speed/displacement bounds")
    # Direct guidance is independent of pursuer positions. Replay only this
    # single aircraft; require exact final state/acceleration and no hard clip.
    attacker = initial[-1].copy()
    acceleration = np.zeros(3, dtype=np.float64)
    for _ in range(steps):
        command = env.attacker_policy(attacker)
        attacker, acceleration = integrate(attacker, command, acceleration, env.dt, env.attacker_limits)
        _require(np.all(attacker[:3] >= env.bounds_low) and np.all(attacker[:3] <= env.bounds_high),
                 "Legacy Direct Attacker replay encountered boundary clipping")
    _require(np.array_equal(attacker, states[-1]) and np.array_equal(acceleration, accelerations[-1]),
             "Legacy Direct Attacker replay does not exactly match environment_state/environment_accelerations")
    return dict(_episode_reward_components=totals, _boundary_clip_counts=np.zeros(4, dtype=np.int64),
                _boundary_clipped=np.zeros(4, dtype=bool), _has_reset=True, _done=False,
                rng_state=state.get("environment_rng_state"), angle_rng_state=state.get("angle_sampling_rng_state"))


def restore_environment(env, state):
    """Restore a checkpoint's same episode, including rewards and sampler RNGs.

    Full snapshots are preferred. Old v09 checkpoints are accepted only after
    proving missing clip counters are zero; no unsupported field is guessed.
    This function modifies the supplied environment, never global RNG streams.
    """
    state = _mapping(state, "training_state")
    for name in ("environment_state", "environment_accelerations", "environment_step_count", "current_episode_angle_degrees"):
        _require(name in state, "Missing " + name)
    raw_states = np.asarray(state["environment_state"])
    _require(raw_states.shape in ((24,), (4, 6)), "environment_state must have shape (24,) or (4, 6)")
    states = _array(raw_states.reshape(4, 6), (4, 6), "environment_state")
    accelerations = _array(state["environment_accelerations"], (4, 3), "environment_accelerations")
    steps = _count(state["environment_step_count"], "environment_step_count")
    _require(steps <= env.max_episode_steps, "environment_step_count exceeds episode limit")
    angle = _number(state["current_episode_angle_degrees"], "current_episode_angle_degrees")
    _require(abs(angle) <= env.half_angle_degrees, "current_episode_angle_degrees exceeds stage range")
    initial = env._reference_initial_states.copy()
    initial[-1, :3] = arrival_position(env.config, angle)
    _require(np.all(states[:, :3] >= env.bounds_low) and np.all(states[:, :3] <= env.bounds_high),
             "environment_state is outside workspace")
    internal = state.get("environment_internal")
    if internal is None:
        internal = _legacy_internal(env, state, states, accelerations, steps, angle, initial)
    else:
        internal = _mapping(internal, "environment_internal")
        _require(internal.get("format_version") == 1, "Unsupported environment_internal format_version")
        _finite_state(internal, "environment_internal")
        for name, expected in (("states", states), ("accelerations", accelerations), ("_initial_states", initial)):
            _require(name in internal and np.array_equal(_array(internal[name], expected.shape, "environment_internal." + name), expected),
                     "environment_internal." + name + " disagrees with checkpoint")
        _require(_count(internal.get("step_count"), "environment_internal.step_count") == steps,
                 "environment_internal.step_count disagrees with checkpoint")
        _require(_number(internal.get("current_angle_degrees"), "environment_internal.current_angle_degrees") == angle,
                 "environment_internal.current_angle_degrees disagrees with checkpoint")
    totals = _mapping(internal.get("_episode_reward_components"), "environment_internal._episode_reward_components")
    _require(set(totals) == set(env._reward_terms), "environment_internal reward components have wrong keys")
    totals = {name: _number(value, "environment_internal._episode_reward_components." + name) for name, value in totals.items()}
    raw_counts = np.asarray(internal.get("_boundary_clip_counts"))
    _require(raw_counts.shape == (4,) and np.issubdtype(raw_counts.dtype, np.integer)
             and np.all(raw_counts >= 0) and np.all(raw_counts <= steps), "Invalid environment_internal._boundary_clip_counts")
    clipped = np.asarray(internal.get("_boundary_clipped"))
    _require(clipped.shape == (4,) and np.issubdtype(clipped.dtype, np.bool_), "Invalid environment_internal._boundary_clipped")
    _require(np.all(raw_counts >= clipped.astype(int)), "Boundary clip counters disagree with latest flags")
    for flag in ("_has_reset", "_done"):
        _require(isinstance(internal.get(flag), (bool, np.bool_)), "Invalid environment_internal." + flag)
    _require(internal["_has_reset"], "Cannot resume environment before reset")
    rng = _generator(internal.get("rng_state"), "environment_rng_state")
    angle_rng = _generator(internal.get("angle_rng_state"), "angle_sampling_rng_state")
    # All validation above precedes mutation of the caller's environment.
    env.states = states
    env.accelerations = accelerations
    env.step_count = steps
    env.current_angle_degrees = angle
    env._initial_states = initial
    env._episode_reward_components = totals
    env._boundary_clip_counts = raw_counts.astype(np.int64, copy=True)
    env._boundary_clipped = clipped.copy()
    env._has_reset = bool(internal["_has_reset"])
    env._done = bool(internal["_done"])
    env.rng, env.angle_rng = rng, angle_rng
    info = env._info()
    terminal = info["captured"] or info["attacker_reached_base"] or steps >= env.max_episode_steps
    _require(env._done == bool(terminal), "environment_internal._done disagrees with termination conditions")
    info["arrival_azimuth_offset_degrees"] = angle
    info["initial_attacker_position"] = initial[-1, :3].copy()
    return env._observations(), info


def _comparable_config(config):
    result = deepcopy(config)
    result.pop("output_dir", None)
    result.pop("continuation", None)
    result.get("training", {}).pop("total_env_steps", None)
    for name in ("max_stage_steps", "stage_step_budgets"):
        result.get("curriculum", {}).pop(name, None)
    return result


def load_resume_source(source_output, config, environment_config):
    """Preflight a stopped run and return its last checkpoint, without a policy."""
    source = Path(source_output).expanduser().resolve()
    summary = _json(source / "summary.json")
    _require(summary.get("status") in ("stopped_stage_budget", "stopped_total_budget"),
             "Resume source must have stopped at stage/total budget")
    stages = summary.get("stages")
    _require(isinstance(stages, list) and bool(stages), "Resume source has no actual stages")
    old_config = _yaml(source / "training_config.yaml")
    _require(_comparable_config(old_config) == _comparable_config(_mapping(config, "config")),
             "Resume configuration differs beyond stage/total budgets, output_dir or continuation")
    old_environment = _yaml(source / "environment_config.yaml")
    _require(old_environment == _mapping(environment_config, "environment_config"), "Resume environment_config differs from source snapshot")
    last_stage = stages[-1]
    index = _count(last_stage.get("stage_index"), "stage_index")
    _require(index >= 1 and index == len(stages), "Source stage_index does not match actual completed stages")
    widths = old_config["curriculum"]["stage_half_angles_degrees"]
    _require(index <= len(widths) and last_stage.get("half_angle_degrees") == widths[index - 1], "Source stage angle mismatch")
    directory = source / "stage_{:02d}_pm{:g}".format(index, widths[index - 1])
    _require(_json(directory / "summary.json") == last_stage, "Source stage/run summaries disagree")
    path = directory / "checkpoints/last.pt"
    _require(path.is_file(), "Resume source last checkpoint is missing: " + str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("format_version") == 1, "Unsupported resume checkpoint format")
    _finite_state(payload, "resume checkpoint")
    _require(_comparable_config(payload.get("config", {})) == _comparable_config(old_config), "Checkpoint config differs from source training config")
    for name in ("actor", "critic", "actor_optimizer", "critic_optimizer", "rng_state", "training_state"):
        _require(isinstance(payload.get(name), dict), "Resume checkpoint is missing " + name)
    for name in ("actor_optimizer", "critic_optimizer"):
        _require("state" in payload[name] and "param_groups" in payload[name], "Missing complete Adam state: " + name)
    state = payload["training_state"]
    for name in ("env_steps", "update", "episodes"):
        _require(_count(state.get(name), name) == summary.get(name), "last.pt " + name + " disagrees with source summary")
    for name in ("stage_index", "stage_env_steps", "stage_updates"):
        _require(_count(state.get(name), name) == last_stage.get(name), "last.pt " + name + " disagrees with stage summary")
    _require(sum(item["stage_env_steps"] for item in stages) == summary["env_steps"], "Source actual step budget mismatch")
    _require(_count(config["training"].get("total_env_steps"), "training.total_env_steps") > summary["env_steps"],
             "Continuation total budget must exceed already-consumed steps")
    stage_budgets = config["curriculum"].get("stage_step_budgets")
    if stage_budgets is None:
        new_budget = config["curriculum"]["max_stage_steps"]
    else:
        _require(isinstance(stage_budgets, list) and len(stage_budgets) == len(widths), "Invalid curriculum.stage_step_budgets")
        new_budget = stage_budgets[index - 1]
    _require(_count(new_budget, "current stage step budget") > state["stage_env_steps"], "Continuation stage budget must exceed consumed stage steps")
    best_path = directory / "checkpoints/best_eval.pt"
    _require(best_path.is_file() and _sha(best_path) == last_stage.get("selected_checkpoint_sha256"), "Source best_eval checkpoint hash mismatch")
    best_summary = _mapping(last_stage.get("selected_evaluation"), "selected_evaluation")
    _require(state.get("best_eval_summary") == best_summary, "last.pt historical best summary disagrees with source stage")
    # Validate reconstructability before the runner creates any new artifacts.
    env = DirectionCurriculumEnv(source / "environment_config.yaml", widths[index - 1], old_config["seed"])
    restore_environment(env, state)
    return dict(source_output=source, summary=summary, payload=payload, checkpoint=path,
                stage_index=index, best_summary=deepcopy(best_summary), best_checkpoint=best_path)


def copy_resume_artifacts(context, output):
    """Copy stage history into a new run; refuse existing stage destinations."""
    source = Path(context["source_output"]).resolve()
    output = Path(output).expanduser().resolve()
    _require(source != output and output not in source.parents,
             "Continuation output must not replace the source run or its parent")
    if source in output.parents:
        relative = output.relative_to(source)
        _require(not relative.parts[0].startswith("stage_"),
                 "Continuation output must not lie inside a source stage directory")
    history = source / "evaluation_history.csv"
    _require(history.is_file(), "Source evaluation_history.csv is missing")
    directories = sorted(path for path in source.glob("stage_*") if path.is_dir())
    _require(bool(directories), "Source has no stage directories")
    destinations = [output / history.name] + [output / path.name for path in directories]
    _require(all(not path.exists() for path in destinations), "Continuation history/stage destination already exists")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(history), str(output / history.name))
    for directory in directories:
        shutil.copytree(str(directory), str(output / directory.name))
