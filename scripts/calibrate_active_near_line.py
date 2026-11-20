"""Calibrate corrected near and line weights on an unchanged replay cohort."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from utils.artifact_paths import resolve_archived_file, resolve_artifact_path

from scripts.offline_near_reward_replay import (
    ROOT, CATEGORIES, read_csv, sha256, write_csv, write_json,
)
from utils.active_near_calibration import score_rows


FIELDS = ("original_return", "original_discounted_return", "near_return", "near_discounted_return",
          "line_return", "line_discounted_return", "total_return", "discounted_return")


def _mean(rows, field):
    return float(np.mean([r[field] for r in rows])) if rows else None


def summarize(rows):
    """Keep real tests, fixed-scene history and stationary controls separate."""
    unique = [r for r in rows if not r["duplicate_of"]]
    combinations = sorted(set((r["beta"], r["k"], r["dataset"], r["cohort"]) for r in unique))
    summaries, categories = [], []
    for beta, k, dataset, cohort in combinations:
        members = [r for r in unique if (r["beta"], r["k"], r["dataset"], r["cohort"]) == (beta, k, dataset, cohort)]
        grouped = {name: [r for r in members if r["category"] == name] for name in CATEGORIES}
        for name, group in grouped.items():
            if not group:
                continue
            category_row = dict(beta=beta, k=k, dataset=dataset, cohort=cohort,
                                category=name, episodes=len(group))
            for field in FIELDS:
                values = np.asarray([r[field] for r in group])
                category_row.update({field + "_mean": float(values.mean()), field + "_std": float(values.std()),
                                     field + "_min": float(values.min()), field + "_max": float(values.max())})
            categories.append(category_row)
        captures = [r for r in members if r["captured"]]
        failures = [r for r in members if not r["captured"]]
        near, moderate = grouped["near_miss"], grouped["moderate_failure"]
        other, clear = grouped["other_failure"], grouped["clear_failure"]
        stationary = {r["scenario_id"]: r for r in unique if (r["beta"], r["k"], r["dataset"], r["cohort"])
                      == (beta, k, dataset, "stationary")}
        closer = [r for r in near if r["scenario_id"] in stationary
                  and r["min_distance"] < stationary[r["scenario_id"]]["min_distance"]] if cohort == "test" else []
        near_mean = _mean(near, "near_return")
        line_mean = _mean(near, "line_return")
        row = dict(beta=beta, k=k, dataset=dataset, cohort=cohort, episodes=len(members),
                   capture_count=len(captures), near_mean=near_mean, line_mean=line_mean,
                   line_to_near_ratio=abs(line_mean) / near_mean if near_mean else None,
                   failure_near_max=max((r["near_return"] for r in failures), default=None),
                   capture_near_mean=_mean(captures, "near_return"),
                   capture_near_max=max((r["near_return"] for r in captures), default=None),
                   success_failure_gap=min(r["total_return"] for r in captures) - max(r["total_return"] for r in failures) if captures and failures else None,
                   discounted_success_failure_gap=min(r["discounted_return"] for r in captures) - max(r["discounted_return"] for r in failures) if captures and failures else None,
                   near_moderate_return_gap=_mean(near, "total_return") - _mean(moderate, "total_return") if near and moderate else None,
                   near_moderate_discounted_gap=_mean(near, "discounted_return") - _mean(moderate, "discounted_return") if near and moderate else None,
                   other_clear_discounted_gap=_mean(other, "discounted_return") - _mean(clear, "discounted_return") if other and clear else None,
                   closer_near_count=len(closer),
                   near_count=len(near),
                   stationary_outrank_near_fraction=float(np.mean([stationary[r["scenario_id"]]["total_return"] > r["total_return"] for r in near])) if near and cohort == "test" else None,
                   stationary_discounted_outrank_near_fraction=float(np.mean([stationary[r["scenario_id"]]["discounted_return"] > r["discounted_return"] for r in near])) if near and cohort == "test" else None,
                   stationary_outrank_closer_fraction=float(np.mean([stationary[r["scenario_id"]]["total_return"] > r["total_return"] for r in closer])) if closer else None,
                   stationary_discounted_outrank_closer_fraction=float(np.mean([stationary[r["scenario_id"]]["discounted_return"] > r["discounted_return"] for r in closer])) if closer else None,
                   missing_categories=",".join(name for name in CATEGORIES if not grouped[name]))
        for field, name in (("total_return", "category_return_order"), ("discounted_return", "category_discounted_order")):
            means = [_mean(grouped[name], field) for name in CATEGORIES if grouped[name]]
            row[name] = all(a > b for a, b in zip(means, means[1:]))
        summaries.append(row)
    return summaries, categories


def recommend(summaries, config, capture_radius):
    """Expose each preference separately and rank feasible positive-line pairs."""
    diagnostics, candidates = [], []
    for beta in config["betas"]:
        for k in config["line_coefficients"]:
            tests = [r for r in summaries if r["beta"] == beta and r["k"] == k and r["cohort"] == "test"
                     and r["dataset"] in config["primary_datasets"]]
            history = [r for r in summaries if r["beta"] == beta and r["k"] == k and r["cohort"] == "history"]
            if len(tests) != len(config["primary_datasets"]):
                raise ValueError("A primary calibration cohort is missing")
            failure_max = max(r["failure_near_max"] for r in tests + history if r["failure_near_max"] is not None)
            diagnostic = dict(beta=beta, k=k, failure_near_max=failure_max,
                failure_near_theoretical_bound=float(beta * np.exp(-capture_radius / config["length_scale_m"])),
                within_failure_near_reference=failure_max <= config["failure_near_budget_reference"],
                success_failure_separated=all(r["success_failure_gap"] > 0 and r["discounted_success_failure_gap"] > 0 for r in tests),
                no_stationary_win_on_closer_near=all(r["stationary_outrank_closer_fraction"] == 0
                    and r["stationary_discounted_outrank_closer_fraction"] == 0 for r in tests),
                category_return_order=all(r["category_return_order"] for r in tests),
                category_discounted_order=all(r["category_discounted_order"] for r in tests),
                max_line_to_near_ratio=max(r["line_to_near_ratio"] for r in tests),
                min_near_moderate_discounted_gap=min(r["near_moderate_discounted_gap"] for r in tests))
            diagnostic["candidate"] = bool(k > 0 and diagnostic["within_failure_near_reference"]
                and diagnostic["success_failure_separated"] and diagnostic["no_stationary_win_on_closer_near"]
                and diagnostic["category_return_order"] and diagnostic["category_discounted_order"])
            diagnostics.append(diagnostic)
            if diagnostic["candidate"]:
                candidates.append(diagnostic)
    # Prefer the lowest near amplification, then the strongest surviving line coefficient.
    candidates.sort(key=lambda r: (r["beta"], -r["k"]))
    winner = candidates[0] if candidates else None
    stronger = next((r for r in candidates if winner and r["k"] == winner["k"] and r["beta"] > winner["beta"]), None)
    rationale = [
        "约80分失败near上限沿用此前讨论的量级偏好，作为本轮离线筛选参考。",
        "分别检查两个真实测试数据组的成功失败间隔、类别均值排序和同角度更近Near miss的静止对照。",
        "候选间优先选较小β，再选满足这些偏好的较大正k，保留几何约束。",
        "经验失败最大值与任意失败回合的理论上界分别记录，提高β会放大理论上界。",
        "这些偏好属于分析选择，不能证明参数最优或训练捕获率会提高。",
    ]
    return dict(status="offline_candidate" if winner else "no_candidate_under_stated_preferences",
                recommended_beta=winner["beta"] if winner else None,
                recommended_k=winner["k"] if winner else None,
                candidates=[dict(beta=r["beta"], k=r["k"]) for r in candidates],
                stronger_comparison=dict(beta=stronger["beta"], k=stronger["k"]) if stronger else None,
                rationale=rationale, diagnostics=diagnostics)


def load_source(config, hashes):
    source = resolve_artifact_path(config["source_analysis"], root=ROOT)
    for filename in ("replay_manifest.json", "analysis_config.yaml", "verification.json", "trajectory_manifest.csv",
                     "episode_reward_replay.csv", "category_statistics.csv", "representatives.csv"):
        path = (source / filename).resolve()
        hashes[str(path)] = sha256(path)
    manifest = json.loads((source / "replay_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["length_scale_m"] != config["length_scale_m"]:
        raise ValueError("A completed same-length-scale source replay is required")
    for name, digest in manifest["source_sha256"].items():
        verified = resolve_archived_file(name, digest, root=ROOT)
        hashes[str(verified)] = digest
    replay_config = yaml.safe_load((source / "analysis_config.yaml").read_text(encoding="utf-8"))
    environment_path = resolve_artifact_path(replay_config["current_root"], root=ROOT) / ("seed_%d" % replay_config["seeds"][0]) / "environment_config.yaml"
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    hashes[str(environment_path.resolve())] = sha256(environment_path)
    manifest["capture_radius_m"] = float(environment["capture"]["radius"])
    manifest["capture_reward"] = float(environment["reward"]["capture_reward"])
    rows = read_csv(source / "episode_reward_replay.csv")
    identities = read_csv(source / "trajectory_manifest.csv")
    if len(rows) != len(identities):
        raise ValueError("Frozen sample count differs from its source mapping")
    for row, identity in zip(rows, identities):
        for field in ("dataset", "cohort", "episode_id", "category", "duplicate_of"):
            if row[field] != identity[field]:
                raise ValueError("Frozen sample identity mismatch")
        for suffix in ("", "_discounted"):
            np.testing.assert_allclose(float(row["old_near_return" + suffix]) - float(row["active_near_return" + suffix]),
                                       float(row["removed_near_return" + suffix]), rtol=0, atol=1e-10)
            np.testing.assert_allclose(float(row["new_with_line_return" + suffix]),
                float(row["original_return" + suffix]) + float(row["active_near_return" + suffix]) + float(row["line_return" + suffix]), rtol=0, atol=1e-10)
    return rows, manifest


def run(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for name in ("betas", "line_coefficients"):
        values = np.asarray(config[name], dtype=float)
        if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(np.diff(values) <= 0)):
            raise ValueError("Calibration grids must be finite, nonnegative and strictly increasing")
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError("Choose a fresh calibration directory " + str(output))
    hashes = {str(config_path): sha256(config_path)}
    source, parent = load_source(config, hashes)
    rows = []
    for beta in config["betas"]:
        for k in config["line_coefficients"]:
            rows.extend(score_rows(source, beta, k, parent["beta"], parent["line_coefficient"]))
    summaries, categories = summarize(rows)
    recommendation = recommend(summaries, config, parent["capture_radius_m"])
    manifest = dict(status="completed", source_rows=len(source), grid_rows=len(rows),
        unique_counts=parent["unique_counts"], source_beta=parent["beta"], source_k=parent["line_coefficient"],
        gamma=parent["gamma"], length_scale_m=config["length_scale_m"], beta_grid=config["betas"],
        capture_radius_m=parent["capture_radius_m"], capture_reward=parent["capture_reward"],
        k_grid=config["line_coefficients"], source_analysis=str((ROOT / config["source_analysis"]).resolve()),
        models_loaded=False, new_inference_episodes=0, training_steps=0,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip())
    output.mkdir(parents=True)
    with (output / "configuration_snapshot.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    write_csv(output / "episode_grid_replay.csv", rows)
    write_csv(output / "combination_summary.csv", summaries)
    write_csv(output / "category_statistics.csv", categories)
    write_csv(output / "candidate_checks.csv", recommendation["diagnostics"])
    write_json(output / "recommendation.json", recommendation)
    from utils.active_near_calibration_reporting import write_results
    manifest["reporting"] = write_results(output, summaries, categories, recommendation, manifest, config)
    for name, digest in hashes.items():
        if sha256(name) != digest:
            raise RuntimeError("Source changed during calibration " + name)
    manifest["source_sha256"] = hashes
    manifest["source_files_unchanged"] = True
    manifest["analysis_code_sha256"] = {name: sha256(ROOT / name) for name in (
        "scripts/calibrate_active_near_line.py", "utils/active_near_calibration.py",
        "utils/active_near_calibration_reporting.py")}
    write_json(output / "replay_manifest.json", manifest)
    write_json(output / "verification.json", dict(source_files_unchanged=True,
        source_rows=len(source), grid_rows=len(rows), models_loaded=False, new_inference_episodes=0, training_steps=0))
    print(json.dumps(dict(output_dir=str(output.resolve()), recommended_beta=recommendation["recommended_beta"],
                          recommended_k=recommendation["recommended_k"], candidates=recommendation["candidates"]), indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/offline_active_near_calibration.yaml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
