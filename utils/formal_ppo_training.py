"""Independent 1v1 PPO training with exact vector budgets and resumable rollouts."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import yaml

from algorithms.ppo import PPOAgent, VectorRolloutBuffer
from envs.interception_env import InterceptionEnv
from utils.artifact_paths import canonicalize_storage_paths, resolve_artifact_path
from utils.experiment import ROOT, set_seed, write_metadata


LOG_FIELDS = {
    "episodes": ("episode", "env_index", "env_episode", "env_steps", "angle_degrees",
                 "initial_attacker_x", "initial_attacker_y", "initial_attacker_z",
                 "episode_return", "episode_length", "captured", "defense_failure",
                 "timeout", "interception_time", "minimum_distance", "final_distance",
                 "final_attacker_base_distance", "boundary_clip_count"),
    "updates": ("update", "env_steps", "rollout_samples", "actor_loss", "value_loss",
                "entropy", "approx_kl", "clip_fraction", "actor_grad_norm", "critic_grad_norm",
                "actor_learning_rate", "critic_learning_rate", "updates_completed_minibatches",
                "epochs_completed", "early_stopped", "log_std_x", "log_std_y", "log_std_z",
                "std_x", "std_y", "std_z"),
    "minibatches": ("update", "env_steps", "epoch", "minibatch", "samples", "actor_loss",
                    "value_loss", "entropy", "approx_kl", "analytic_kl", "clip_fraction",
                    "actor_grad_norm", "critic_grad_norm", "actor_learning_rate", "critic_learning_rate"),
    "epochs": ("update", "env_steps", "epoch", "minibatches", "samples", "clip_fraction",
               "approx_kl", "early_stopped"),
    "validation": ("env_steps", "update", "last_update_env_steps", "pending_rollout_samples",
                   "capture_rate", "defense_failure_rate", "timeout_rate", "average_interception_time",
                   "average_min_distance", "average_return", "consecutive_qualifying_validations",
                   "stability_reached", "new_best", "relative_output"),
    "normalization": ("env_steps", "env_index", "episode_step", "angle_degrees", "field_index",
                      "raw_value", "normalized_value"),
}


SHAPING_EPISODE_FIELDS = ("original_return", "near_shaping_return", "best_distance_so_far",
                          "reward_absolute_distance_cost", "reward_survival",
                          "reward_capture", "reward_attack_failure")
LINE_EPISODE_FIELDS = ("line_cost_return", "line_segment_distance", "return_without_line_cost")
LINE_VALIDATION_FIELDS = ("average_original_return", "average_near_shaping_return",
                          "average_line_cost_return", "average_return_without_line_cost",
                          "average_line_segment_distance")
MOTION_NEAR_EPISODE_FIELDS = ("near_shaping_uncorrected_return", "near_shaping_removed_return",
                              "near_counterfactual_distance")
MOTION_NEAR_VALIDATION_FIELDS = tuple("average_" + field for field in (
    "original_return", "near_shaping_return") + MOTION_NEAR_EPISODE_FIELDS)


def reward_episode_fields(environment):
    """Select optional reward columns from the saved environment configuration."""
    reward = environment["reward"]
    fields = SHAPING_EPISODE_FIELDS if reward.get("best_so_far_shaping") is not None else ()
    if reward.get("line_cost_coefficient") is not None:
        if not fields:
            fields = ("original_return",)
        fields += LINE_EPISODE_FIELDS
    shaping = reward.get("best_so_far_shaping")
    if shaping is not None and shaping.get("mode") == "motion_attributed":
        fields += MOTION_NEAR_EPISODE_FIELDS
    return fields


def shaping_episode_metrics(info):
    """Expose optional cumulative rewards using the environment's actual sums."""
    components = info.get("episode_reward_components", {})
    original_fields = ("absolute_distance_cost", "survival", "capture", "attack_failure")
    result = {}
    if "near_shaping" in components:
        result.update(original_return=float(sum(components[name] for name in original_fields)),
                      near_shaping_return=float(components["near_shaping"]),
                      best_distance_so_far=float(info["best_distance_so_far"]),
                      **{"reward_" + name: float(components[name]) for name in original_fields})
    if "line_cost" in components:
        result.setdefault("original_return", float(sum(components[name] for name in original_fields)))
        result.update(line_cost_return=float(components["line_cost"]),
                      line_segment_distance=float(info["line_segment_distance"]),
                      return_without_line_cost=float(info["episode_return"] - components["line_cost"]))
    if "episode_near_shaping_uncorrected" in info:
        result.update(near_shaping_uncorrected_return=float(info["episode_near_shaping_uncorrected"]),
                      near_shaping_removed_return=float(info["episode_near_shaping_removed"]),
                      near_counterfactual_distance=float(info["near_counterfactual_distance"]))
    return result


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_integer(value, name):
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             name + " must be a positive integer")
    return value


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_json_value(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def global_rng_state():
    return dict(python=random.getstate(), numpy=deepcopy(np.random.get_state()),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                torch_num_threads=torch.get_num_threads(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark)


def restore_global_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    torch.set_num_threads(state["torch_num_threads"])
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]


def _state_hash(value):
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode("ascii"))
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                add(child)
        else:
            digest.update((type(item).__name__ + repr(item)).encode("utf-8"))

    add(value)
    return digest.hexdigest()


class IndependentEnvironments:
    """Synchronous vector collection; every member owns an independent RNG/state."""

    def __init__(self, environment_config, num_envs, seed):
        children = np.random.SeedSequence(seed).spawn(num_envs)
        self.seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
        _require(len(set(self.seeds)) == num_envs, "Derived environment seeds collided")
        self.envs = [InterceptionEnv(deepcopy(environment_config), seed=child_seed)
                     for child_seed in self.seeds]

    def reset(self):
        reset = [env.reset(seed=seed) for env, seed in zip(self.envs, self.seeds)]
        return np.stack([item[0] for item in reset]), [item[1] for item in reset]

    def step(self, actions):
        _require(np.asarray(actions).shape == (len(self.envs), 3), "Vector action shape differs")
        transitions = [env.step(action) for env, action in zip(self.envs, actions)]
        _require(not any(item[3] for item in transitions), "Formal timeout must be a true terminal")
        return (np.stack([item[0] for item in transitions]),
                np.asarray([item[1] for item in transitions], dtype=np.float64),
                np.asarray([item[2] for item in transitions], dtype=bool),
                [item[4] for item in transitions])

    def state_dict(self):
        return dict(seeds=list(self.seeds), environments=[env.get_state() for env in self.envs])

    def load_state_dict(self, state):
        _require(len(state["environments"]) == len(self.envs), "Environment count differs on restore")
        self.seeds = list(state["seeds"])
        for env, saved in zip(self.envs, state["environments"]):
            env.set_state(deepcopy(saved))


def load_configuration(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    environment_path = Path(config["env_config"])
    if not environment_path.is_absolute():
        environment_path = path.parent / environment_path
    config["environment_config"] = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    config["_config_source"] = str(path)
    config["_environment_source"] = str(environment_path.resolve())
    return config


def validate_config(config):
    training = config["training"]
    num_envs = _positive_integer(training["num_envs"], "training.num_envs")
    total = _positive_integer(training["total_env_steps"], "training.total_env_steps")
    interval = _positive_integer(training["evaluation_interval_steps"], "training.evaluation_interval_steps")
    for name in ("rollout_steps", "validation_episodes", "stability_consecutive", "checkpoint_interval_updates"):
        _positive_integer(training[name], "training." + name)
    decay_steps = _positive_integer(training.get("learning_rate_decay_steps", total),
                                   "training.learning_rate_decay_steps")
    _require(decay_steps <= total, "Learning-rate decay horizon must be within the total budget")
    _require(total % num_envs == 0 and interval % num_envs == 0,
             "Total budget and evaluation interval must align with complete vector steps")
    threshold = training["stability_threshold"]
    _require(np.isfinite(threshold) and 0 <= threshold <= 1, "Invalid stability threshold")
    environment = config.get("environment_config")
    _require(isinstance(environment, dict), "environment_config must contain the resolved environment YAML")
    normalization = environment.get("normalization", {})
    _require(normalization.get("confirmed") is True,
             "Formal training blocked: the 16 physical normalization divisors require confirmation")
    _require(normalization.get("acceleration_kind") == "applied",
             "Formal training requires post-jerk applied observation acceleration")
    scales = np.asarray(normalization.get("scales"), dtype=float)
    _require(scales.shape == (16,) and np.isfinite(scales).all() and np.all(scales > 0),
             "Formal training blocked: supply all 16 confirmed positive normalization divisors")
    _require(not config.get("curriculum") and not config.get("warm_start"),
             "Formal PPO uses independent fresh runs or explicit exact resume")
    return config


def _episode_start(env, info):
    return dict(episode_return=0.0, episode_length=0,
                minimum_distance=float(info["min_distance"]),
                angle_degrees=float(env.angle_degrees),
                initial_attacker_position=env.states[1, :3].copy())


def _record_violations(logs, info, env_steps, env_index):
    event = info.get("normalization_violations", [])
    events = [event] if isinstance(event, dict) else event
    for violation in events:
        for index, raw, normalized in zip(violation.get("field_indices", []),
                                          violation.get("raw_values", []), violation.get("normalized_values", [])):
            logs.append(dict(env_steps=env_steps, env_index=env_index,
                             episode_step=violation.get("step"), angle_degrees=violation.get("angle_degrees"),
                             field_index=int(index), raw_value=float(raw), normalized_value=float(normalized)))


def _atomic_checkpoint(path, bundle, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(overwrite or not path.exists(), "Refusing to overwrite checkpoint: " + str(path))
    temporary = path.with_name(path.name + ".partial")
    _require(not temporary.exists(), "Partial checkpoint already exists: " + str(temporary))
    torch.save(bundle, temporary)
    temporary.replace(path)


def _copy_prior_checkpoints(source_bundle, output):
    source_output = resolve_artifact_path(source_bundle["output_dir"])
    step = source_bundle["training_state"]["env_steps"]
    source_checkpoints = source_output / "checkpoints"
    for path in source_checkpoints.glob("*.pt"):
        if path.name == "initial.pt":
            shutil.copyfile(path, output / "checkpoints" / path.name)
        elif path.stem.startswith(("checkpoint_", "evaluated_")):
            if int(path.stem.rsplit("_", 1)[1]) <= step:
                shutil.copyfile(path, output / "checkpoints" / path.name)
    for directory in (source_output / "validation").glob("steps_*"):
        if int(directory.name.rsplit("_", 1)[1]) <= step:
            shutil.copytree(directory, output / "validation" / directory.name)


def train(config, output_dir, seed, resume_checkpoint=None, stop_at_steps=None,
          evaluation_fn=None, post_training_fn=None, allow_budget_extension=False):
    """Train one run, optionally pausing at an exact vector-step safe point.

    A pause retains the original total budget and scheduler horizon. Resume
    creates a new directory and restores the unfinished on-policy buffer before
    collecting another transition. Validation at a sample milestone does not
    shorten the current rollout or perform an extra PPO update.
    """
    config = deepcopy(config)
    validate_config(config)
    _require(isinstance(allow_budget_extension, bool), "allow_budget_extension must be boolean")
    _require(not allow_budget_extension or resume_checkpoint is not None,
             "Budget extension requires a completed source checkpoint")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "Invalid training seed")
    output = Path(output_dir).resolve()
    _require(not output.exists(), "Use a new output directory: " + str(output))
    training, environment = config["training"], config["environment_config"]
    num_envs, total = training["num_envs"], training["total_env_steps"]
    stop = total if stop_at_steps is None else stop_at_steps
    _require(isinstance(stop, int) and 0 < stop <= total and stop % num_envs == 0,
             "Pause step must be a positive complete vector step within the total budget")
    config["seed"] = seed
    config["output_dir"] = str(output)
    if evaluation_fn is None:
        from utils.formal_ppo_evaluation import evaluate_angles
        evaluation_fn = evaluate_angles
    set_seed(seed, config.get("torch_num_threads", 1))
    agent = PPOAgent(config)
    vector = IndependentEnvironments(environment, num_envs, seed)
    observations, initial_info = vector.reset()
    log_fields = dict(LOG_FIELDS)
    log_fields["episodes"] += reward_episode_fields(environment)
    if environment["reward"].get("line_cost_coefficient") is not None:
        log_fields["validation"] += LINE_VALIDATION_FIELDS
    shaping = environment["reward"].get("best_so_far_shaping")
    if shaping is not None and shaping.get("mode") == "motion_attributed":
        log_fields["validation"] += tuple(field for field in MOTION_NEAR_VALIDATION_FIELDS
                                          if field not in log_fields["validation"])
    buffer = VectorRolloutBuffer(training["rollout_steps"], num_envs, 16, 3)
    state = dict(env_steps=0, update=0, episodes=0, last_update_env_steps=0,
                 rollout_target_steps=None, next_evaluation_steps=training["evaluation_interval_steps"],
                 consecutive_qualifying_validations=0, stability_reached=False,
                 first_stable_env_steps=None, best_key=None, best_env_steps=None,
                 env_episode_counts=[0] * num_envs,
                 episode_accumulators=[_episode_start(env, info) for env, info in zip(vector.envs, initial_info)],
                 logs={name: [] for name in LOG_FIELDS})
    best_checkpoint = None
    original = None
    if resume_checkpoint is None:
        for index, info in enumerate(initial_info):
            _record_violations(state["logs"]["normalization"], info, 0, index)
    if resume_checkpoint is not None:
        resume_checkpoint = resolve_artifact_path(resume_checkpoint)
        original = torch.load(resume_checkpoint, map_location=config.get("device", "cpu"), weights_only=False)
        _require(original.get("format_version") == 1 and original.get("kind") == "formal_single_ppo",
                 "Unsupported formal PPO checkpoint")
        expected_config = deepcopy(original["config"])
        actual_config = deepcopy(config)
        for mapping in (expected_config, actual_config):
            for key in ("output_dir", "_config_source", "_environment_source"):
                mapping.pop(key, None)
        if allow_budget_extension:
            previous_budget = original["config"]["training"]["total_env_steps"]
            source_state = original["training_state"]
            source_scheduler = original["agent"]["scheduler"]
            _require(total > previous_budget, "Budget extension must increase the completed source budget")
            _require(source_state["env_steps"] == previous_budget
                     and source_state["last_update_env_steps"] == previous_budget
                     and source_state["rollout_buffer"]["size"] == 0
                     and source_state["rollout_target_steps"] is None
                     and source_scheduler["environment_steps"] == previous_budget,
                     "Budget extension requires a completed source budget with an empty rollout")
            _require(training.get("learning_rate_decay_steps") == source_scheduler["total_env_steps"],
                     "Budget extension must preserve the source learning-rate decay horizon")
            expected_config["training"]["total_env_steps"] = total
            expected_config["training"]["learning_rate_decay_steps"] = source_scheduler["total_env_steps"]
            expected_config["experiment"]["output_root"] = actual_config["experiment"]["output_root"]
        _require(canonicalize_storage_paths(expected_config) == canonicalize_storage_paths(actual_config)
                 and original["seed"] == seed,
                 "Resume changes learning, environment, seed or budget settings")
        agent.load_state_dict(original["agent"], allow_budget_extension=allow_budget_extension)
        vector.load_state_dict(original["training_state"]["vector_environments"])
        observations = original["training_state"]["observations"].copy()
        buffer.load_state_dict(original["training_state"]["rollout_buffer"])
        state = deepcopy(original["training_state"])
        state.pop("vector_environments")
        state.pop("observations")
        state.pop("rollout_buffer")
        best_checkpoint = deepcopy(original.get("best_checkpoint"))
        if best_checkpoint is None and original.get("is_best_checkpoint"):
            best_checkpoint = deepcopy(original)
        _require(state["env_steps"] <= stop, "Requested pause precedes the restored checkpoint")
        restore_global_rng(original["rng_state"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "validation").mkdir()
    for filename, mapping in (("training_config.yaml", config), ("environment_config.yaml", environment)):
        (output / filename).write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
    metadata = write_metadata(output / "metadata.json", config, environment,
                              config_path=config.get("_config_source", ROOT / "configs/ppo_formal_v1.yaml"),
                              env_config_path=config.get("_environment_source", ROOT / "configs/interception_1v1.yaml"),
                              output_dir=output, kind="formal_single_ppo",
                              extra=dict(environment_seeds=vector.seeds, num_environments=num_envs,
                                         aggregate_budget=total, resumed_from=None if resume_checkpoint is None else str(Path(resume_checkpoint).resolve())))
    if original is not None:
        _copy_prior_checkpoints(original, output)
        if best_checkpoint is not None:
            _atomic_checkpoint(output / "checkpoints/best.pt", best_checkpoint)

    def capture(include_best=True, is_best=False):
        training_state = deepcopy(state)
        training_state.update(vector_environments=vector.state_dict(), observations=observations.copy(),
                              rollout_buffer=buffer.state_dict())
        return dict(format_version=1, kind="formal_single_ppo", config=deepcopy(config),
                    environment_config=deepcopy(environment), seed=seed, output_dir=str(output),
                    agent=deepcopy(agent.state_dict()), training_state=training_state,
                    rng_state=global_rng_state(), metadata=deepcopy(metadata), is_best_checkpoint=is_best,
                    best_checkpoint=deepcopy(best_checkpoint) if include_best else None)

    if original is None:
        _atomic_checkpoint(output / "checkpoints/initial.pt", capture())
    else:
        _atomic_checkpoint(output / "checkpoints/resume_initial.pt", capture())
    streams, writers = {}, {}
    try:
        for name, fields in log_fields.items():
            streams[name] = (output / (name + ".csv")).open("x", encoding="utf-8", newline="")
            writers[name] = csv.DictWriter(streams[name], fieldnames=fields, extrasaction="ignore")
            writers[name].writeheader()
            writers[name].writerows(state["logs"][name])
            streams[name].flush()

        def log(name, row):
            state["logs"][name].append(deepcopy(row))
            writers[name].writerow(row)
            streams[name].flush()

        def validate_current():
            nonlocal best_checkpoint
            step = state["env_steps"]
            eval_path = output / "validation" / ("steps_{:06d}".format(step))
            initialization = environment["initialization"]
            angle_range = initialization.get("angle_range_degrees", initialization.get("attacker_angle_range_degrees"))
            _require(np.asarray(angle_range).shape == (2,), "Configured arrival-angle range must have two endpoints")
            angles = np.linspace(float(angle_range[0]), float(angle_range[1]), training["validation_episodes"])
            saved_rng, modes = global_rng_state(), (agent.actor.training, agent.critic.training)
            before = _state_hash(agent.state_dict())
            try:
                agent.actor.eval()
                agent.critic.eval()
                with torch.inference_mode():
                    summary = evaluation_fn(agent, deepcopy(environment), angles,
                                            config["experiment"]["validation_seed"], eval_path)
            finally:
                restore_global_rng(saved_rng)
                agent.actor.train(modes[0])
                agent.critic.train(modes[1])
            _require(_state_hash(agent.state_dict()) == before, "Validation changed model/optimizer/scheduler state")
            for key in ("capture_rate", "defense_failure_rate", "timeout_rate"):
                _require(np.isfinite(summary[key]) and 0 <= summary[key] <= 1, "Invalid validation " + key)
            _require(np.isclose(summary["capture_rate"] + summary["defense_failure_rate"] + summary["timeout_rate"], 1),
                     "Validation outcome rates must sum to one")
            achieved = summary["capture_rate"] >= training["stability_threshold"]
            state["consecutive_qualifying_validations"] = state["consecutive_qualifying_validations"] + 1 if achieved else 0
            if state["consecutive_qualifying_validations"] >= training["stability_consecutive"]:
                if not state["stability_reached"]:
                    state["first_stable_env_steps"] = step
                state["stability_reached"] = True
            time = summary["average_interception_time"]
            maximum_time = environment["environment"]["dt"] * environment["environment"]["max_episode_steps"]
            key = (float(summary["capture_rate"]), -float(summary["defense_failure_rate"]),
                   -(float(time) if time is not None else maximum_time + environment["environment"]["dt"]),
                   -float(summary["average_min_distance"]))
            improved = state["best_key"] is None or key > tuple(state["best_key"])
            if improved:
                state["best_key"], state["best_env_steps"] = key, step
            row = dict(env_steps=step, update=state["update"], last_update_env_steps=state["last_update_env_steps"],
                       pending_rollout_samples=buffer.size * num_envs, **summary,
                       consecutive_qualifying_validations=state["consecutive_qualifying_validations"],
                       stability_reached=state["stability_reached"], new_best=improved,
                       relative_output=eval_path.relative_to(output).as_posix())
            log("validation", row)
            if step == state["next_evaluation_steps"]:
                state["next_evaluation_steps"] += training["evaluation_interval_steps"]
            eval_path.mkdir(parents=True, exist_ok=True)
            if not (eval_path / "summary.json").exists():
                _write_json(eval_path / "summary.json", summary)
            if improved:
                best_checkpoint = capture(include_best=False, is_best=True)
                _atomic_checkpoint(output / "checkpoints/best.pt", best_checkpoint, overwrite=True)
            _atomic_checkpoint(output / "checkpoints/evaluated_{:06d}.pt".format(step), capture())
            print("seed={} validation_steps={} capture_rate={:.3f} stable_streak={} new_best={}".format(
                seed, step, summary["capture_rate"], state["consecutive_qualifying_validations"], improved), flush=True)

        while state["env_steps"] < stop:
            updated = False
            if state["rollout_target_steps"] is None:
                state["rollout_target_steps"] = min(buffer.capacity, (total - state["env_steps"]) // num_envs)
            decision = agent.act(observations)
            next_observations, rewards, terminated, infos = vector.step(decision["actions"])
            next_values = agent.value(next_observations)
            next_values = np.where(terminated, 0.0, next_values)
            buffer.add(observations, decision["raw_actions"], decision["log_probs"], decision["means"],
                       decision["log_stds"], rewards, decision["values"], next_values, terminated)
            state["env_steps"] += num_envs
            observations = next_observations.copy()
            for index, (env, reward, done, info) in enumerate(zip(vector.envs, rewards, terminated, infos)):
                accumulator = state["episode_accumulators"][index]
                accumulator["episode_return"] += float(reward)
                accumulator["episode_length"] += 1
                accumulator["minimum_distance"] = min(accumulator["minimum_distance"], float(info["min_distance"]))
                violation_rows = []
                _record_violations(violation_rows, info, state["env_steps"], index)
                for violation in violation_rows:
                    log("normalization", violation)
                if done:
                    state["episodes"] += 1
                    state["env_episode_counts"][index] += 1
                    initial_position = accumulator["initial_attacker_position"]
                    captured = bool(info["captured"])
                    log("episodes", dict(episode=state["episodes"], env_index=index,
                        env_episode=state["env_episode_counts"][index], env_steps=state["env_steps"],
                        angle_degrees=accumulator["angle_degrees"],
                        **dict(zip(("initial_attacker_x", "initial_attacker_y", "initial_attacker_z"), initial_position)),
                        episode_return=accumulator["episode_return"], episode_length=accumulator["episode_length"],
                        captured=captured, defense_failure=bool(info["enemy_attack_success"]),
                        timeout=bool(info["timeout"]), interception_time=accumulator["episode_length"] * env.dt if captured else None,
                        minimum_distance=accumulator["minimum_distance"], final_distance=float(info["min_distance"]),
                        final_attacker_base_distance=float(info["distance_to_base"]),
                        boundary_clip_count=int(info.get("boundary_clip_count", 0)),
                        **shaping_episode_metrics(info)))
                    observations[index], reset_info = env.reset()
                    state["episode_accumulators"][index] = _episode_start(env, reset_info)
                    reset_violations = []
                    _record_violations(reset_violations, reset_info, state["env_steps"], index)
                    for violation in reset_violations:
                        log("normalization", violation)
            if buffer.size == state["rollout_target_steps"]:
                rollout_samples = buffer.size * num_envs
                buffer.compute_returns(config["ppo"]["gamma"], config["ppo"]["gae_lambda"])
                metrics = agent.update(buffer, env_steps=state["env_steps"])
                state["update"] += 1
                state["last_update_env_steps"] = state["env_steps"]
                common = dict(update=state["update"], env_steps=state["env_steps"])
                logs = agent.actor.log_std.detach().cpu().numpy()
                stds = np.exp(logs)
                update_row = dict(common, rollout_samples=rollout_samples,
                                  **{k: v for k, v in metrics.items() if k not in ("minibatches", "epochs")},
                                  **{"log_std_" + axis: float(logs[i]) for i, axis in enumerate("xyz")},
                                  **{"std_" + axis: float(stds[i]) for i, axis in enumerate("xyz")})
                log("updates", update_row)
                for row in metrics["minibatches"]:
                    log("minibatches", dict(common, **row))
                for row in metrics["epochs"]:
                    log("epochs", dict(common, **row))
                buffer.reset()
                state["rollout_target_steps"] = None
                updated = True
                print("seed={} steps={}/{} update={} kl={:.6f} early_stop={}".format(
                    seed, state["env_steps"], total, state["update"], metrics["approx_kl"], metrics["early_stopped"]), flush=True)
            if state["env_steps"] == state["next_evaluation_steps"]:
                validate_current()
            if updated and state["update"] % training["checkpoint_interval_updates"] == 0:
                _atomic_checkpoint(output / "checkpoints/checkpoint_{:06d}.pt".format(state["env_steps"]), capture())
        if state["env_steps"] == total:
            if not state["logs"]["validation"] or state["logs"]["validation"][-1]["env_steps"] != total:
                validate_current()
            _atomic_checkpoint(output / "checkpoints/final.pt", capture())
            status = "completed"
        else:
            _atomic_checkpoint(output / "checkpoints/paused_{:06d}.pt".format(state["env_steps"]), capture())
            status = "paused"
    finally:
        for stream in streams.values():
            stream.close()
    summary = dict(status=status, seed=seed, env_steps=state["env_steps"], updates=state["update"],
                   episodes=state["episodes"], num_envs=num_envs, total_budget=total,
                   best_env_steps=state["best_env_steps"], best_validation_key=state["best_key"],
                   stability_reached=state["stability_reached"], first_stable_env_steps=state["first_stable_env_steps"],
                   output_dir=str(output), resumed_from=None if resume_checkpoint is None else str(Path(resume_checkpoint).resolve()),
                   pending_rollout_samples=buffer.size * num_envs)
    _write_json(output / "summary.json", summary)
    if status == "completed" and post_training_fn is not None:
        post_training_fn(output, deepcopy(config), deepcopy(summary))
    return summary
