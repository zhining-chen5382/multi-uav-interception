"""Calibrate a proposed line-segment cost using frozen recorded trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from utils.artifact_paths import resolve_artifact_path

from scripts.offline_near_reward_replay import (
    ROOT, CATEGORIES, category, discounted_return, read_csv, sha256, write_csv, write_json,
    _pairwise_win_rate,
)
from utils.line_cost_replay import (
    episode_components, line_cost, segment_distances, stationary_counterfactual,
)


def _track(path, hashes):
    path = resolve_artifact_path(path, root=ROOT)
    hashes.setdefault(str(path), sha256(path))
    return path


def _load_yaml(path, hashes):
    return yaml.safe_load(_track(path, hashes).read_text(encoding="utf-8"))


def _episode(path, environment, identity, hashes):
    path = _track(path, hashes)
    with np.load(path, allow_pickle=False) as data:
        result = episode_components(data["states"], data["time"], environment)
        saved_rewards = data["rewards"]
        metadata = json.loads(data["metadata_json"].item())
        np.testing.assert_allclose(result["components"]["near_shaping"],
                                   data["near_shaping"], rtol=0, atol=1e-10)
    error = float(np.max(np.abs(result["components"]["total"] - saved_rewards)))
    np.testing.assert_allclose(result["components"]["total"], saved_rewards, rtol=0, atol=1e-10)
    np.testing.assert_allclose(saved_rewards.sum(), metadata["episode_return"], rtol=0, atol=1e-8)
    for field in ("captured", "defense_failure", "timeout"):
        if result[field] != metadata[field]:
            raise ValueError("Saved outcome disagrees with trajectory " + str(path))
    result.update(identity, scenario_id=int(metadata["scenario_id"]),
                  environment_seed=int(metadata["seed"]), angle_degrees=float(metadata["angle_degrees"]),
                  trajectory=str(path), trajectory_sha256=hashes[str(path)],
                  max_reward_reconstruction_error=error)
    return result


def _finish_episode(episode, environment, seen):
    states, time = episode["states"], episode["time"]
    positions = states[:, :, :3]
    base = np.asarray(environment["mission"]["base_position"], dtype=float)
    episode["base_position"] = base.copy()
    error = segment_distances(positions[:, 0], positions[:, 1], base)
    episode["segment_distances"] = error
    episode["line_unit_cost"] = line_cost(error, time, 1.0)
    episode["category"] = category(episode["captured"], float(episode["distances"].min()))
    digest = hashlib.sha256(states.tobytes() + time.tobytes()).hexdigest()
    key = (episode["cohort"], digest)
    episode["duplicate_of"] = seen.get(key, "")
    seen.setdefault(key, episode["episode_id"])
    episode["content_sha256"] = digest
    direction = positions[:, 1] - base
    denominator = np.sum(direction * direction, axis=1)
    fraction = np.zeros(len(states))
    np.divide(np.sum((positions[:, 0] - base) * direction, axis=1), denominator,
              out=fraction, where=denominator > 0)
    line_nearest = base + fraction[:, None] * direction
    infinite_distance = np.linalg.norm(positions[:, 0] - line_nearest, axis=1)
    steps = np.diff(time)
    closest = int(np.argmin(episode["distances"]))
    episode.update(
        segment_integral=float(-episode["line_unit_cost"].sum()),
        segment_mean_distance=float(-episode["line_unit_cost"].sum() / time[-1]),
        segment_max_distance=float(error.max()),
        infinite_line_integral=float(np.dot(infinite_distance[1:], steps)),
        before_base_integral=float(np.dot(error[1:] * (fraction[1:] < 0), steps)),
        beyond_attacker_integral=float(np.dot(error[1:] * (fraction[1:] > 1), steps)),
        after_closest_integral=float(-episode["line_unit_cost"][closest:].sum()),
        final_attacker_base_distance=float(np.linalg.norm(positions[-1, 1] - base)))
    return episode


def load_episodes(config, hashes):
    root = resolve_artifact_path(config["source_root"], root=ROOT)
    experiment = _load_yaml(root / "experiment_config.yaml", hashes)
    gamma = float(experiment["ppo"]["gamma"])
    episodes, seen, environments = [], {}, {}
    for seed in config["seeds"]:
        directory = root / ("seed_%d" % seed)
        environment = _load_yaml(directory / "environment_config.yaml", hashes)
        if environments and environment != next(iter(environments.values())):
            raise ValueError("Calibration cohorts must use identical environment configurations")
        environments[seed] = environment
        for name in ("envs/interception_env.py", "algorithms/ppo/agent.py",
                     "utils/formal_ppo_training.py", "configs/ppo_reward_ab.yaml",
                     "configs/interception_1v1_best_so_far.yaml"):
            _track(ROOT / name, hashes)
        for model, folder in (("best", "final_test"), ("final", "final_model_test")):
            evaluation = directory / folder
            _track(directory / "checkpoints" / (model + ".pt"), hashes)
            rows = read_csv(_track(evaluation / "episodes.csv", hashes))
            if len(rows) != experiment["experiment"]["test_episodes"]:
                raise ValueError("Incomplete frozen test cohort")
            step = int(json.loads((directory / "summary.json").read_text(encoding="utf-8"))[
                "best_env_steps"]) if model == "best" else experiment["training"]["total_env_steps"]
            _track(directory / "summary.json", hashes)
            for row in rows:
                identity = dict(episode_id="test_seed%d_%s_scene%s" % (seed, model, row["scenario_id"]),
                                cohort="test", training_seed=seed, model=model, checkpoint_step=step)
                episode = _episode(evaluation / row["trajectory"], environment, identity, hashes)
                episodes.append(_finish_episode(episode, environment, seen))
        history = directory / config["postprocessing"][seed]
        rows = read_csv(_track(history / "trajectory_evolution/checkpoint_metrics.csv", hashes))
        for index, row in enumerate(rows):
            identity = dict(episode_id="history_seed%d_record%03d" % (seed, index),
                            cohort="history", training_seed=seed, model="checkpoint_history",
                            checkpoint_step=int(row["step"]))
            episode = _episode(history / row["trajectory"], environment, identity, hashes)
            episodes.append(_finish_episode(episode, environment, seen))
    source_seed = config["stationary_source_seed"]
    environment = environments[source_seed]
    sources = [episode for episode in episodes if episode["cohort"] == "test"
               and episode["training_seed"] == source_seed and episode["model"] == "final"]
    reference = {episode["scenario_id"]: episode for episode in sources}
    if len(reference) != len(sources):
        raise ValueError("Stationary reference scenario IDs must be unique")
    for episode in episodes:
        if episode["cohort"] != "test":
            continue
        paired = reference[episode["scenario_id"]]
        if (episode["environment_seed"] != paired["environment_seed"]
                or episode["angle_degrees"] != paired["angle_degrees"]):
            raise ValueError("Test scenes differ from their stationary reference")
        count = len(episode["states"])
        np.testing.assert_array_equal(episode["states"][:, 1], paired["states"][:count, 1])
        np.testing.assert_array_equal(episode["time"], paired["time"][:count])
    for source in sources:
        episode = stationary_counterfactual(source["states"], source["time"], environment)
        episode.update(episode_id="stationary_scene%d" % source["scenario_id"], cohort="stationary",
                       training_seed="", model="stationary_counterfactual", checkpoint_step="",
                       scenario_id=source["scenario_id"], environment_seed=source["environment_seed"],
                       angle_degrees=source["angle_degrees"], trajectory=source["trajectory"],
                       trajectory_sha256=source["trajectory_sha256"], max_reward_reconstruction_error=0.0)
        episodes.append(_finish_episode(episode, environment, seen))
    return episodes, gamma, environments


def replay(episodes, coefficients, gamma):
    coefficients = np.asarray(coefficients, dtype=float)
    if (coefficients.ndim != 1 or len(coefficients) == 0 or not np.isfinite(coefficients).all()
            or np.any(coefficients < 0) or len(set(coefficients)) != len(coefficients)
            or coefficients[0] != 0 or np.any(np.diff(coefficients) <= 0)):
        raise ValueError("Coefficient grid must start at zero and increase strictly")
    rows = []
    identity_fields = ("episode_id", "cohort", "training_seed", "model", "checkpoint_step",
                       "scenario_id", "environment_seed", "angle_degrees", "category", "duplicate_of")
    geometry_fields = ("segment_integral", "segment_mean_distance", "segment_max_distance",
                       "infinite_line_integral", "before_base_integral", "beyond_attacker_integral",
                       "after_closest_integral", "final_attacker_base_distance")
    for episode in episodes:
        components = episode["components"]
        baseline = components["total"]
        original_return = float(components["original"].sum())
        near_return = float(components["near_shaping"].sum())
        baseline_discounted = discounted_return(baseline, gamma)
        for coefficient in coefficients:
            penalty = line_cost(episode["segment_distances"], episode["time"], coefficient)
            row = {key: episode[key] for key in identity_fields + geometry_fields}
            row.update(coefficient=float(coefficient), episode_length=len(baseline),
                       event_time=float(episode["time"][-1]),
                       min_distance=float(episode["distances"].min()), original_return=original_return,
                       near_return=near_return, baseline_return=float(baseline.sum()),
                       baseline_discounted_return=baseline_discounted,
                       line_penalty=float(penalty.sum()), line_cost_magnitude=float(-penalty.sum()),
                       discounted_line_penalty=discounted_return(penalty, gamma),
                       new_return=float((baseline + penalty).sum()),
                       new_discounted_return=discounted_return(baseline + penalty, gamma))
            rows.append(row)
    return rows


def summarize(rows, config):
    unique = [row for row in rows if not row["duplicate_of"]]
    fields = ("original_return", "near_return", "baseline_return", "new_return",
              "baseline_discounted_return", "new_discounted_return", "line_cost_magnitude",
              "segment_integral", "infinite_line_integral", "before_base_integral",
              "beyond_attacker_integral", "after_closest_integral")
    category_rows, pairwise, summaries = [], [], []
    for coefficient in config["coefficients"]:
        for cohort in ("test", "history", "stationary"):
            selected = [row for row in unique if row["coefficient"] == coefficient and row["cohort"] == cohort]
            groups = {name: [row for row in selected if row["category"] == name] for name in CATEGORIES}
            for name, group in groups.items():
                if not group:
                    continue
                summary = dict(cohort=cohort, coefficient=coefficient, category=name, episodes=len(group))
                for field in fields:
                    values = np.asarray([row[field] for row in group])
                    summary.update({field + "_mean": float(values.mean()), field + "_min": float(values.min()),
                                    field + "_max": float(values.max()), field + "_std": float(values.std())})
                category_rows.append(summary)
            for higher, lower in zip(CATEGORIES, CATEGORIES[1:]):
                if not groups[higher] or not groups[lower]:
                    continue
                for field in ("baseline_return", "new_return", "baseline_discounted_return", "new_discounted_return"):
                    pairwise.append(dict(cohort=cohort, coefficient=coefficient, return_type=field,
                        higher_category=higher, lower_category=lower,
                        pairwise_win_rate=_pairwise_win_rate([r[field] for r in groups[higher]],
                                                            [r[field] for r in groups[lower]])))
        selected = [row for row in unique if row["coefficient"] == coefficient and row["cohort"] == "test"]
        stationary = {row["scenario_id"]: row for row in unique if row["coefficient"] == coefficient
                      and row["cohort"] == "stationary"}
        groups = {name: [row for row in selected if row["category"] == name] for name in CATEGORIES}
        result = dict(coefficient=coefficient, unique_test_episodes=len(selected),
                      capture_count=len(groups["capture"]))
        for name, group in groups.items():
            if not group:
                raise ValueError("Calibration requires every requested trajectory category")
            result[name + "_mean_cost"] = float(np.mean([row["line_cost_magnitude"] for row in group]))
            result[name + "_mean_new_return"] = float(np.mean([row["new_return"] for row in group]))
            result[name + "_mean_new_discounted_return"] = float(np.mean([row["new_discounted_return"] for row in group]))
        captures = groups["capture"]
        failures = [row for row in selected if row["category"] != "capture"]
        near = groups["near_miss"]
        wins = [stationary[row["scenario_id"]]["new_return"] > row["new_return"] for row in near]
        discounted_wins = [stationary[row["scenario_id"]]["new_discounted_return"] > row["new_discounted_return"] for row in near]
        margins = [row["new_return"] - stationary[row["scenario_id"]]["new_return"] for row in near]
        closer_near = [row for row in near if row["min_distance"] < stationary[row["scenario_id"]]["min_distance"]]
        closer_wins = [stationary[row["scenario_id"]]["new_return"] > row["new_return"] for row in closer_near]
        closer_discounted_wins = [stationary[row["scenario_id"]]["new_discounted_return"] > row["new_discounted_return"] for row in closer_near]
        result.update(capture_max_cost=max(row["line_cost_magnitude"] for row in captures),
            capture_min_new_return=min(row["new_return"] for row in captures),
            max_failure_new_return=max(row["new_return"] for row in failures),
            capture_failure_gap=min(row["new_return"] for row in captures) - max(row["new_return"] for row in failures),
            discounted_capture_failure_mean_gap=result["capture_mean_new_discounted_return"] - max(
                result[name + "_mean_new_discounted_return"] for name in CATEGORIES[1:]),
            stationary_mean_new_return=float(np.mean([row["new_return"] for row in stationary.values()])),
            stationary_mean_cost=float(np.mean([row["line_cost_magnitude"] for row in stationary.values()])),
            stationary_near_miss_outrank_fraction=float(np.mean(wins)),
            stationary_near_miss_discounted_outrank_fraction=float(np.mean(discounted_wins)),
            closer_near_miss_count=len(closer_near),
            stationary_closer_near_outrank_fraction=float(np.mean(closer_wins)) if closer_near else None,
            stationary_closer_near_discounted_outrank_fraction=float(np.mean(closer_discounted_wins)) if closer_near else None,
            paired_near_minus_stationary_mean=float(np.mean(margins)),
            paired_near_minus_stationary_integral=float(np.mean([
                row["segment_integral"] - stationary[row["scenario_id"]]["segment_integral"] for row in near])))
        for field, label in (("mean_new_return", "five_class_mean_order"),
                             ("mean_new_discounted_return", "discounted_five_class_mean_order")):
            means = [result[name + "_" + field] for name in CATEGORIES]
            result[label] = bool(all(a > b for a, b in zip(means, means[1:])))
        summaries.append(result)
    baseline = summaries[0]["stationary_near_miss_outrank_fraction"]
    checks = config["calibration_checks"]
    for row in summaries:
        row["stationary_near_miss_outrank_increase"] = row["stationary_near_miss_outrank_fraction"] - baseline
        row["calibration_candidate"] = bool(
            row["coefficient"] > 0
            and row["capture_mean_cost"] <= checks["capture_mean_cost_max"]
            and row["capture_max_cost"] <= checks["capture_max_cost_max"]
            and row["clear_failure_mean_cost"] >= checks["clear_failure_mean_cost_min"]
            and row["discounted_capture_failure_mean_gap"] > checks["capture_failure_discounted_mean_gap_min"]
            and row["stationary_near_miss_outrank_increase"] <= checks["stationary_near_miss_outrank_increase_max"])
    candidates = [row["coefficient"] for row in summaries if row["calibration_candidate"]]
    recommendation = dict(status="candidate" if candidates else "no_candidate",
                          recommended_coefficient=min(candidates) if candidates else None,
                          candidate_coefficients=candidates, checks=checks,
                          paired_baseline_margin=summaries[0]["paired_near_minus_stationary_mean"],
                          paired_integral_gap=summaries[0]["paired_near_minus_stationary_integral"])
    return summaries, category_rows, pairwise, recommendation


def run(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError("Choose a new analysis output directory " + str(output))
    hashes = {}
    _track(config_path, hashes)
    episodes, gamma, environments = load_episodes(config, hashes)
    rows = replay(episodes, config["coefficients"], gamma)
    summaries, categories, pairwise, recommendation = summarize(rows, config)
    counts = {cohort: sum(episode["cohort"] == cohort for episode in episodes)
              for cohort in ("test", "history", "stationary")}
    unique_counts = {cohort: sum(episode["cohort"] == cohort and not episode["duplicate_of"] for episode in episodes)
                     for cohort in counts}
    manifest = dict(status="completed", gamma=gamma, source_counts=counts, unique_counts=unique_counts,
                    geometry=config["geometry"], sampling=config["sampling"],
                    coefficients=config["coefficients"], coefficient_units=config["coefficient_units"],
                    max_reward_reconstruction_error=max(episode["max_reward_reconstruction_error"] for episode in episodes),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
                    stationary_provenance="P0 fixed at configured initial position and zero velocity; saved Direct Attacker path; first terminal event re-evaluated",
                    recommendation=recommendation)
    output.mkdir(parents=True)
    with (output / "analysis_config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    manifest_fields = ("episode_id", "cohort", "training_seed", "model", "checkpoint_step", "scenario_id",
                       "environment_seed", "angle_degrees", "category", "trajectory", "trajectory_sha256",
                       "content_sha256", "duplicate_of")
    write_csv(output / "trajectory_manifest.csv", [{key: episode[key] for key in manifest_fields} for episode in episodes])
    write_csv(output / "episode_reward_replay.csv", rows)
    write_csv(output / "coefficient_summary.csv", summaries)
    write_csv(output / "category_statistics.csv", categories)
    write_csv(output / "pairwise_rankings.csv", pairwise)
    write_json(output / "recommendation.json", recommendation)
    from utils.line_cost_replay_reporting import write_results
    write_results(output, episodes, rows, summaries, categories, pairwise, recommendation, manifest, config)
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise RuntimeError("Source file changed during analysis " + path)
    manifest["source_sha256"] = hashes
    manifest["source_files_unchanged"] = True
    manifest["analysis_code_sha256"] = {
        name: sha256(ROOT / name) for name in (
            "scripts/offline_line_cost_replay.py", "utils/line_cost_replay.py", "utils/line_cost_replay_reporting.py",
            "scripts/offline_near_reward_replay.py", "scripts/offline_best_so_far_reward_replay.py")}
    write_json(output / "replay_manifest.json", manifest)
    print(json.dumps(dict(output_dir=str(output.resolve()), source_counts=counts, unique_counts=unique_counts,
                          recommendation=recommendation), ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/offline_line_cost_replay.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
