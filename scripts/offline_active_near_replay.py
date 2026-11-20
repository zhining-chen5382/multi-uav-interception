"""Replay motion-attributed record-distance shaping on saved trajectories."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from utils.artifact_paths import resolve_artifact_path

from scripts.offline_line_cost_replay import load_episodes, _finish_episode, _track
from scripts.offline_near_reward_replay import (
    ROOT, CATEGORIES, discounted_return, read_csv, sha256, write_csv,
    write_json, _pairwise_win_rate,
)
from utils.active_near_replay import active_near_shaping
from utils.line_cost_replay import episode_components, line_cost, segment_distances, stationary_counterfactual


RETURN_FIELDS = (
    "original_return", "old_near_return", "active_near_return", "removed_near_return",
    "line_return", "old_without_line_return", "new_without_line_return",
    "old_with_line_return", "new_with_line_return",
)
ALL_RETURN_FIELDS = RETURN_FIELDS + tuple(name + "_discounted" for name in RETURN_FIELDS)
COMPARISON_FIELDS = (
    "old_without_line_return", "new_without_line_return",
    "old_with_line_return", "new_with_line_return",
    "old_without_line_return_discounted", "new_without_line_return_discounted",
    "old_with_line_return_discounted", "new_with_line_return_discounted",
)


def _yaml(path, hashes):
    return yaml.safe_load(_track(path, hashes).read_text(encoding="utf-8"))


def load_reference(config, hashes):
    """Reconstruct the exact previous sample IDs, categories and deduplication."""
    previous = resolve_artifact_path(config["previous_analysis"], root=ROOT)
    parent_manifest = json.loads(_track(previous / "replay_manifest.json", hashes).read_text(encoding="utf-8"))
    parent_config = _yaml(previous / "analysis_config.yaml", hashes)
    frozen = read_csv(_track(previous / "trajectory_manifest.csv", hashes))
    episodes, gamma, environments = load_episodes(parent_config, hashes)
    if len(frozen) != len(episodes):
        raise ValueError("Previous trajectory count changed")
    fields = ("episode_id", "cohort", "category", "trajectory_sha256", "content_sha256", "duplicate_of")
    for episode, expected in zip(episodes, frozen):
        for field in fields:
            if str(episode[field]) != expected[field]:
                raise ValueError("Previous sample identity changed: " + episode["episode_id"])
        episode["dataset"] = "reference_B"
    for cohort, count in parent_manifest["unique_counts"].items():
        actual = sum(e["cohort"] == cohort and not e["duplicate_of"] for e in episodes)
        if actual != count:
            raise ValueError("Previous unique cohort count changed")
    representatives = read_csv(_track(previous / "representatives.csv", hashes))
    return episodes, gamma, environments, representatives


def _current_episode(path, environment, identity, hashes):
    path = _track(path, hashes)
    with np.load(path, allow_pickle=False) as data:
        episode = episode_components(data["states"], data["time"], environment)
        metadata = json.loads(data["metadata_json"].item())
        geometry = segment_distances(data["states"][:, 0, :3], data["states"][:, 1, :3],
                                     environment["mission"]["base_position"])
        np.testing.assert_allclose(geometry, data["line_distances"], rtol=0, atol=1e-10)
        penalty = line_cost(geometry, data["time"], environment["reward"]["line_cost_coefficient"])
        expected = episode["components"]["total"] + penalty
        error = float(np.max(np.abs(expected - data["rewards"])))
        np.testing.assert_allclose(expected, data["rewards"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(episode["components"]["near_shaping"], data["near_shaping"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(penalty, data["line_cost"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(expected.sum(), metadata["episode_return"], rtol=0, atol=1e-8)
    for field in ("captured", "defense_failure", "timeout"):
        if episode[field] != metadata[field]:
            raise ValueError("Saved outcome disagrees with geometry")
    episode.update(identity, dataset="current_k0p1", scenario_id=int(metadata["scenario_id"]),
                   environment_seed=int(metadata["seed"]), angle_degrees=float(metadata["angle_degrees"]),
                   trajectory=str(path), trajectory_sha256=hashes[str(path)],
                   max_reward_reconstruction_error=error)
    return episode


def load_current(config, hashes, reference, gamma):
    root = resolve_artifact_path(config["current_root"], root=ROOT)
    experiment = _yaml(root / "experiment_config.yaml", hashes)
    if float(experiment["ppo"]["gamma"]) != gamma:
        raise ValueError("Training discount factors differ")
    episodes, seen, environments = [], {}, {}
    scenes = {e["scenario_id"]: e for e in reference if e["cohort"] == "stationary"}
    for seed in config["seeds"]:
        run = root / ("seed_%d" % seed)
        environment = _yaml(run / "environment_config.yaml", hashes)
        environments[seed] = environment
        summary = json.loads(_track(run / "summary.json", hashes).read_text(encoding="utf-8"))
        if environment["reward"]["line_cost_coefficient"] != config["line_coefficient"]:
            raise ValueError("Analysis line coefficient differs from current experiment")
        for model, folder in (("best", "final_test"), ("final", "final_model_test")):
            evaluation = run / folder
            _track(run / "checkpoints" / (model + ".pt"), hashes)
            rows = read_csv(_track(evaluation / "episodes.csv", hashes))
            if len(rows) != experiment["experiment"]["test_episodes"]:
                raise ValueError("Incomplete current evaluation")
            step = summary["best_env_steps"] if model == "best" else summary["env_steps"]
            for row in rows:
                identity = dict(episode_id="current_seed%d_%s_scene%s" % (seed, model, row["scenario_id"]),
                                cohort="test", training_seed=seed, model=model, checkpoint_step=step)
                episode = _current_episode(evaluation / row["trajectory"], environment, identity, hashes)
                if str(episode["scenario_id"]) != row["scenario_id"]:
                    raise ValueError("Current index and trajectory scene IDs differ")
                paired = scenes[episode["scenario_id"]]
                if (episode["environment_seed"] != paired["environment_seed"]
                        or episode["angle_degrees"] != paired["angle_degrees"]):
                    raise ValueError("Current and previous test scenes differ")
                count = min(len(episode["states"]), len(paired["states"]))
                np.testing.assert_array_equal(episode["states"][:count, 1], paired["states"][:count, 1])
                episodes.append(_finish_episode(episode, environment, seen))
    source_seed = config["stationary_source_seed"]
    environment = environments[source_seed]
    sources = [e for e in episodes if e["training_seed"] == source_seed and e["model"] == "final"]
    for source in sources:
        episode = stationary_counterfactual(source["states"], source["time"], environment)
        episode.update(dataset="current_k0p1", episode_id="current_stationary_scene%d" % source["scenario_id"],
                       cohort="stationary", training_seed="", model="stationary_counterfactual", checkpoint_step="",
                       scenario_id=source["scenario_id"], environment_seed=source["environment_seed"],
                       angle_degrees=source["angle_degrees"], trajectory=source["trajectory"],
                       trajectory_sha256=source["trajectory_sha256"], max_reward_reconstruction_error=0.0)
        episodes.append(_finish_episode(episode, environment, seen))
    return episodes, environments


def replay_episode(episode, config, gamma):
    """Replace only near shaping, with line-inclusive and line-free views."""
    values = active_near_shaping(episode["states"], config["length_scale_m"], config["beta"])
    old = episode["components"]["near_shaping"]
    np.testing.assert_allclose(values["original_near"], old, rtol=0, atol=1e-10)
    active = values["active_near"]
    if np.any(active < 0) or np.any(active > old + 1e-10):
        raise ValueError("Corrected shaping violates the original reward bound")
    if episode["cohort"] == "stationary" and np.any(active != 0):
        raise ValueError("Stationary shaping must be exactly zero")
    original = episode["components"]["original"]
    penalty = line_cost(episode["segment_distances"], episode["time"], config["line_coefficient"])
    components = dict(original_return=original, old_near_return=old,
                      active_near_return=active, removed_near_return=old - active,
                      line_return=penalty, old_without_line_return=original + old,
                      new_without_line_return=original + active,
                      old_with_line_return=original + old + penalty,
                      new_with_line_return=original + active + penalty)
    fields = ("episode_id", "dataset", "cohort", "training_seed", "model", "checkpoint_step",
              "scenario_id", "environment_seed", "angle_degrees", "category", "duplicate_of")
    row = {name: episode[name] for name in fields}
    row.update(min_distance=float(values["distances"].min()), event_time=float(episode["time"][-1]),
               episode_length=len(old), captured=episode["captured"],
               defense_failure=episode["defense_failure"], timeout=episode["timeout"],
               line_coefficient=config["line_coefficient"],
               positive_near_steps=int(np.count_nonzero(old > 0)),
               active_near_steps=int(np.count_nonzero(active > 0)))
    for field, series in components.items():
        row[field] = float(series.sum())
        row[field + "_discounted"] = discounted_return(series, gamma)
    row["removed_fraction"] = row["removed_near_return"] / row["old_near_return"] if row["old_near_return"] else 0.0
    values.update(time=episode["time"], states=episode["states"],
                  original=original, line=penalty, episode_id=episode["episode_id"], row=row)
    return row, values


def summarize(rows):
    unique = [r for r in rows if not r["duplicate_of"]]
    categories, pairwise, stationary_rows, seed_rows = [], [], [], []
    groups = sorted(set((r["dataset"], r["cohort"]) for r in unique))
    for dataset, cohort in groups:
        members = [r for r in unique if r["dataset"] == dataset and r["cohort"] == cohort]
        for name in CATEGORIES:
            selected = [r for r in members if r["category"] == name]
            if not selected:
                continue
            summary = dict(dataset=dataset, cohort=cohort, category=name, episodes=len(selected))
            for field in ALL_RETURN_FIELDS:
                values = np.asarray([r[field] for r in selected])
                summary.update({field + "_mean": float(values.mean()), field + "_min": float(values.min()),
                                field + "_max": float(values.max()), field + "_std": float(values.std())})
            categories.append(summary)
        for higher, lower in zip(CATEGORIES, CATEGORIES[1:]):
            high = [r for r in members if r["category"] == higher]
            low = [r for r in members if r["category"] == lower]
            if high and low:
                for field in COMPARISON_FIELDS:
                    pairwise.append(dict(dataset=dataset, cohort=cohort, return_field=field,
                        higher_category=higher, lower_category=lower,
                        win_rate=_pairwise_win_rate([r[field] for r in high], [r[field] for r in low])))
        if cohort != "test":
            continue
        stationary = {r["scenario_id"]: r for r in unique if r["dataset"] == dataset and r["cohort"] == "stationary"}
        near = [r for r in members if r["category"] == "near_miss"]
        closer = [r for r in near if r["min_distance"] < stationary[r["scenario_id"]]["min_distance"]]
        for field in COMPARISON_FIELDS:
            stationary_rows.append(dict(dataset=dataset, return_field=field,
                stationary_mean_return=float(np.mean([r[field] for r in stationary.values()])),
                near_count=len(near),
                stationary_outrank_fraction=float(np.mean([stationary[r["scenario_id"]][field] > r[field] for r in near])),
                closer_near_count=len(closer),
                stationary_outrank_closer_fraction=float(np.mean([stationary[r["scenario_id"]][field] > r[field] for r in closer])) if closer else None))
        model_members = [r for r in rows if r["dataset"] == dataset and r["cohort"] == cohort]
        for seed, model in sorted(set((r["training_seed"], r["model"]) for r in model_members)):
            selected = [r for r in model_members if r["training_seed"] == seed and r["model"] == model]
            result = dict(dataset=dataset, training_seed=seed, model=model, episodes=len(selected),
                          captures=sum(r["captured"] for r in selected))
            for field in ALL_RETURN_FIELDS:
                result[field + "_mean"] = float(np.mean([r[field] for r in selected]))
            seed_rows.append(result)
    return categories, pairwise, stationary_rows, seed_rows


def _representative_ids(episodes, previous, config):
    selected = {("reference_B", r["episode_id"]) for r in previous}
    for name in config["current_representative_categories"]:
        members = [e for e in episodes if e["dataset"] == "current_k0p1" and e["cohort"] == "test"
                   and not e["duplicate_of"] and e["category"] == name]
        if members:
            median = float(np.median([e["distances"].min() for e in members]))
            episode = min(members, key=lambda e: (abs(float(e["distances"].min()) - median), e["episode_id"]))
            selected.add((episode["dataset"], episode["episode_id"]))
    return selected


def run(config_path):
    config_path = Path(config_path).resolve()
    hashes = {}
    config = _yaml(config_path, hashes)
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError("Choose a fresh replay directory " + str(output))
    episodes, gamma, reference_environments, previous = load_reference(config, hashes)
    current, current_environments = load_current(config, hashes, episodes, gamma)
    episodes.extend(current)
    for environment in list(reference_environments.values()) + list(current_environments.values()):
        expected = environment["reward"]["best_so_far_shaping"]
        if (expected["beta"] != config["beta"] or expected["length_scale_m"] != config["length_scale_m"]):
            raise ValueError("Replay coefficients differ from source shaping")
    ids = _representative_ids(episodes, previous, config)
    rows, representatives = [], []
    for episode in episodes:
        row, values = replay_episode(episode, config, gamma)
        rows.append(row)
        if (episode["dataset"], episode["episode_id"]) in ids:
            representatives.append(values)
    category_rows, pairwise, stationary_rows, seed_rows = summarize(rows)
    counts, unique = {}, {}
    for dataset, cohort in sorted(set((e["dataset"], e["cohort"]) for e in episodes)):
        key = dataset + "/" + cohort
        counts[key] = sum(e["dataset"] == dataset and e["cohort"] == cohort for e in episodes)
        unique[key] = sum(e["dataset"] == dataset and e["cohort"] == cohort and not e["duplicate_of"] for e in episodes)
    manifest = dict(status="completed", gamma=gamma, beta=config["beta"], length_scale_m=config["length_scale_m"],
        line_coefficient=config["line_coefficient"], source_counts=counts, unique_counts=unique,
        max_reward_reconstruction_error=max(e["max_reward_reconstruction_error"] for e in episodes),
        models_loaded=False, new_inference_episodes=0, training_steps=0,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
        baseline="Freeze the pursuer position for one recorded transition; reuse actual next attacker position",
        attribution="Pursuer displacement including inertial motion; positive clipping and actual record-distance history",
        stationary_max_active_reward=max(r["active_near_return"] for r in rows if r["cohort"] == "stationary"),
        max_active_minus_old=max(r["active_near_return"] - r["old_near_return"] for r in rows),
        test_scene_limitation="These paired test angles have already been used for reward development")
    output.mkdir(parents=True)
    with (output / "analysis_config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    fields = ("dataset", "episode_id", "cohort", "training_seed", "model", "checkpoint_step", "scenario_id",
              "environment_seed", "angle_degrees", "category", "trajectory", "trajectory_sha256", "content_sha256", "duplicate_of")
    write_csv(output / "trajectory_manifest.csv", [{name: e[name] for name in fields} for e in episodes])
    for filename, values in (("episode_reward_replay.csv", rows), ("category_statistics.csv", category_rows),
                             ("stationary_comparison.csv", stationary_rows),
                             ("seed_model_summary.csv", seed_rows)):
        write_csv(output / filename, values)
    with (output / "pairwise_rankings.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "cohort", "return_field", "higher_category", "lower_category", "win_rate"))
        writer.writeheader()
        writer.writerows(pairwise)
    step_rows = []
    for values in representatives:
        for step in range(len(values["original_near"])):
            step_rows.append(dict(dataset=values["row"]["dataset"], episode_id=values["episode_id"], step=step,
                time=float(values["time"][step + 1]), distance_before=float(values["distances"][step]),
                distance_after=float(values["distances"][step + 1]),
                best_before=float(values["best_distances"][step]), best_after=float(values["best_distances"][step + 1]),
                counterfactual_distance=float(values["counterfactual_distances"][step]),
                original_near=float(values["original_near"][step]), active_near=float(values["active_near"][step]),
                removed_near=float(values["removed_near"][step]), original_reward=float(values["original"][step]),
                line_reward=float(values["line"][step])))
    write_csv(output / "representative_step_components.csv", step_rows)
    from utils.active_near_replay_reporting import write_results
    manifest["reporting"] = write_results(output, rows, category_rows, pairwise, stationary_rows,
                                          representatives, manifest, config)
    plot_lookup = {(r["dataset"], r["episode_id"]): r["plot"]
                   for r in manifest["reporting"]["representatives"]}
    sources = {(e["dataset"], e["episode_id"]): e for e in episodes}
    representative_rows = []
    for values in representatives:
        row = dict(values["row"])
        key = (row["dataset"], row["episode_id"])
        source = sources[key]
        row.update(trajectory=source["trajectory"], trajectory_sha256=source["trajectory_sha256"],
                   component_plot=plot_lookup[key],
                   selection="Previous fixed representative" if row["dataset"] == "reference_B"
                   else "Unique current class median minimum distance")
        representative_rows.append(row)
    write_csv(output / "representatives.csv", representative_rows)
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise RuntimeError("Source changed during replay " + path)
    manifest["source_sha256"] = hashes
    manifest["source_files_unchanged"] = True
    manifest["analysis_code_sha256"] = {name: sha256(ROOT / name) for name in (
        "scripts/offline_active_near_replay.py", "utils/active_near_replay.py",
        "utils/active_near_replay_reporting.py", "utils/line_cost_replay.py",
        "scripts/offline_line_cost_replay.py", "scripts/offline_near_reward_replay.py")}
    write_json(output / "verification.json", dict(
        source_files_unchanged=True, stationary_max_active_reward=manifest["stationary_max_active_reward"],
        per_step_active_between_zero_and_original=True, previous_sample_identity_and_categories_preserved=True,
        real_trajectories_and_termination_reused=True,
        max_reward_reconstruction_error=manifest["max_reward_reconstruction_error"],
        models_loaded=False, new_inference_episodes=0, training_steps=0))
    write_json(output / "replay_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in (
        "source_counts", "unique_counts", "stationary_max_active_reward", "source_files_unchanged")}, indent=2))
    print(str(output.resolve()))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/offline_active_near_replay.yaml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
