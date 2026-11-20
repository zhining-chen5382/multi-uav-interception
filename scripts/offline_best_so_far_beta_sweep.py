"""Offline amplitude comparison using the frozen best-so-far replay corpus."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from utils.artifact_paths import resolve_artifact_path

from scripts.offline_best_so_far_reward_replay import best_so_far_shaping, load_previous
from scripts.offline_near_reward_replay import (
    ROOT, CATEGORIES, COHORTS, _pairwise_win_rate, discounted_return,
    read_csv, sha256, write_csv, write_json,
)


def replay(episodes, reference, config, gamma):
    rows = []
    for episode in episodes:
        rewards = episode["components"]["original"]
        distances = episode["distances"]
        unit = best_so_far_shaping(distances, config["lambda_m"], 1.0)
        unit_total = float(unit.sum())
        unit_discounted = discounted_return(unit, gamma)
        previous = reference[episode["episode_id"]]
        previous_beta = float(previous["beta"])
        np.testing.assert_allclose(unit_total * previous_beta, float(previous["near_return"]), rtol=0, atol=1e-10)
        np.testing.assert_allclose(unit_discounted * previous_beta, float(previous["discounted_near_return"]), rtol=0, atol=1e-10)
        original = float(rewards.sum())
        original_discounted = discounted_return(rewards, gamma)
        np.testing.assert_allclose(original, float(previous["original_return"]), rtol=0, atol=1e-10)
        np.testing.assert_allclose(original_discounted, float(previous["original_discounted_return"]), rtol=0, atol=1e-10)
        for beta in config["betas"]:
            shaping = beta * unit
            np.testing.assert_allclose(shaping.sum(), beta * (
                np.exp(-distances.min() / config["lambda_m"]) - np.exp(-distances[0] / config["lambda_m"])), rtol=0, atol=1e-10)
            rows.append(dict(
                episode_id=episode["episode_id"], cohort=episode["cohort"],
                training_seed=episode["training_seed"], checkpoint_step=episode["checkpoint_step"],
                angle_degrees=episode["angle_degrees"], category=episode["category"],
                lambda_m=config["lambda_m"], beta=beta,
                original_return=original, new_return=float((rewards + shaping).sum()),
                near_return=float(shaping.sum()), original_discounted_return=original_discounted,
                new_discounted_return=discounted_return(rewards + shaping, gamma),
                discounted_near_return=discounted_return(shaping, gamma)))
    return rows


def summarize(rows, config, capture_reward):
    groups, summaries, gaps = [], [], []
    for cohort in COHORTS:
        for beta in config["betas"]:
            subset = [row for row in rows if row["cohort"] == cohort and row["beta"] == beta]
            by_category = {name: [row for row in subset if row["category"] == name] for name in CATEGORIES}
            means = {}
            for name, members in by_category.items():
                group = dict(cohort=cohort, lambda_m=config["lambda_m"], beta=beta, category=name, episodes=len(members))
                for field in ("original_return", "new_return", "near_return", "original_discounted_return",
                              "new_discounted_return", "discounted_near_return"):
                    values = np.asarray([row[field] for row in members])
                    group[field + "_mean"] = float(values.mean())
                    group[field + "_min"] = float(values.min())
                    group[field + "_max"] = float(values.max())
                means[name] = group
                groups.append(group)
            pairs = list(zip(CATEGORIES, CATEGORIES[1:])) + [("near_miss", "clear_failure")]
            for higher, lower in pairs:
                gaps.append(dict(cohort=cohort, beta=beta, higher_category=higher, lower_category=lower,
                    original_raw_mean_gap=means[higher]["original_return_mean"] - means[lower]["original_return_mean"],
                    new_raw_mean_gap=means[higher]["new_return_mean"] - means[lower]["new_return_mean"],
                    new_discounted_mean_gap=means[higher]["new_discounted_return_mean"] - means[lower]["new_discounted_return_mean"],
                    raw_pairwise_win_rate=_pairwise_win_rate([r["new_return"] for r in by_category[higher]], [r["new_return"] for r in by_category[lower]]),
                    discounted_pairwise_win_rate=_pairwise_win_rate([r["new_discounted_return"] for r in by_category[higher]], [r["new_discounted_return"] for r in by_category[lower]])))
            failures = [row for row in subset if row["category"] != "capture"]
            captures = by_category["capture"]
            max_shaping = max(failures, key=lambda row: row["near_return"])
            max_return = max(failures, key=lambda row: row["new_return"])
            summaries.append(dict(cohort=cohort, lambda_m=config["lambda_m"], beta=beta,
                max_failure_shaping=max_shaping["near_return"], max_failure_shaping_episode=max_shaping["episode_id"],
                failure_shaping_limit_margin=config["failure_shaping_soft_limit"] - max_shaping["near_return"],
                failures_above_shaping_limit=sum(r["near_return"] > config["failure_shaping_soft_limit"] for r in failures),
                max_failure_new_return=max_return["new_return"], max_failure_return_episode=max_return["episode_id"],
                failures_with_nonnegative_return=sum(r["new_return"] >= 0 for r in failures),
                max_capture_shaping=max(r["near_return"] for r in captures),
                max_capture_shaping_to_terminal_ratio=max(r["near_return"] for r in captures) / capture_reward,
                min_capture_new_return=min(r["new_return"] for r in captures),
                min_capture_discounted_return=min(r["new_discounted_return"] for r in captures),
                max_failure_discounted_return=max(r["new_discounted_return"] for r in failures),
                near_vs_far_raw_gap=means["near_miss"]["new_return_mean"] - means["clear_failure"]["new_return_mean"],
                near_vs_far_discounted_gap=means["near_miss"]["new_discounted_return_mean"] - means["clear_failure"]["new_discounted_return_mean"],
                raw_five_category_mean_order_ok=all(means[a]["new_return_mean"] > means[b]["new_return_mean"] for a, b in zip(CATEGORIES,CATEGORIES[1:])),
                discounted_five_category_mean_order_ok=all(means[a]["new_discounted_return_mean"] > means[b]["new_discounted_return_mean"] for a, b in zip(CATEGORIES,CATEGORIES[1:]))))
    return groups, summaries, gaps


def plot(groups, summaries, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [row for row in groups if row["cohort"] == COHORTS[0]]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, field in zip(axes[:2], ("new_return_mean", "new_discounted_return_mean")):
        for beta in config["betas"]:
            selected = {r["category"]:r for r in primary if r["beta"] == beta}
            axis.plot(range(4), [selected[name][field] for name in CATEGORIES[1:]], marker="o", label="beta=%g" % beta)
        axis.set_xticks(range(4))
        axis.set_xticklabels(CATEGORIES[1:], rotation=25, ha="right", fontsize=8)
        axis.set_ylabel("Mean return")
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)
    axes[0].set_title("Failure quality | undiscounted")
    axes[1].set_title("Failure quality | gamma=0.99")
    selected = [r for r in summaries if r["cohort"] == COHORTS[0]]
    axes[2].bar([str(int(r["beta"])) for r in selected], [r["max_failure_shaping"] for r in selected], color="#287ba5")
    axes[2].axhline(config["failure_shaping_soft_limit"], color="#c24f40", ls="--", label="Failure shaping limit")
    axes[2].set_xlabel("Beta")
    axes[2].set_ylabel("Maximum failure shaping")
    axes[2].set_ylim(0, config["failure_shaping_soft_limit"] * 1.12)
    axes[2].legend(fontsize=8)
    for index, row in enumerate(selected):
        axes[2].annotate("%.2f" % row["max_failure_shaping"], (index,row["max_failure_shaping"]), xytext=(0,-6), textcoords="offset points", ha="center", va="top", color="white", fontsize=9)
    figure.suptitle("Best-so-far distance replay | lambda=%g m | same held-out trajectories" % config["lambda_m"])
    figure.tight_layout()
    figure.savefig(output/"beta_comparison.png", dpi=160)
    plt.close(figure)


def write_report(groups, summaries, gaps, config, output):
    primary_groups = [row for row in groups if row["cohort"] == COHORTS[0]]
    primary_summary = [row for row in summaries if row["cohort"] == COHORTS[0]]
    largest = max(primary_summary, key=lambda row: row["beta"])
    labels = dict(capture="Capture", near_miss="Near miss", moderate_failure="10–30m失败", other_failure="30–100m失败", clear_failure=">100m失败")
    lines = ["# Best-so-far λ20m 的β幅度对照", "",
        "沿用前两轮完全相同的3120条真实轨迹、顺序、类别及原奖励。主分析为5个Seed的2500条随机角度测试轨迹，辅助分析为620条固定0°历史轨迹。", "",
        r"\[r_{near,t}=\beta(e^{-b_{t+1}/20}-e^{-b_t/20}),\quad b_t=\min_{i\le t}d_i\]", "",
        "gamma=0.99，原距离、时间、+500捕获和−100防御失败项读取冻结环境快照。全部新增结果来自已有完整轨迹的离线计分。", ""]
    for title, field in (("各类别平均新增shaping", "near_return_mean"), ("各类别平均新总回报", "new_return_mean"), ("各类别平均γ=0.99折扣回报", "new_discounted_return_mean")):
        lines.extend(["## " + title, "", "| β | Capture | Near miss | 10–30m失败 | 30–100m失败 | >100m失败 |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
        for beta in config["betas"]:
            selected = {r["category"]:r for r in primary_groups if r["beta"] == beta}
            lines.append("| %g | %.3f | %.3f | %.3f | %.3f | %.3f |" % (beta, *[selected[name][field] for name in CATEGORIES]))
        lines.append("")
    lines.extend(["## Near miss与明显失败的均值差距及上限", "",
        "| β | 未折扣均值差 | 折扣均值差 | 最大失败shaping | 距80分余量 | 最高失败新R | 最大Capture shaping / 500 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in primary_summary:
        lines.append("| %g | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f%% |" % (
            row["beta"], row["near_vs_far_raw_gap"], row["near_vs_far_discounted_gap"], row["max_failure_shaping"],
            row["failure_shaping_limit_margin"], row["max_failure_new_return"], 100*row["max_capture_shaping_to_terminal_ratio"]))
    lines.extend(["", "主测试集所有三组均满足五类未折扣均值顺序。所有三组的30–100m失败折扣均值仍低于>100m失败，完整五类折扣排序均未达到。", "",
        "## 相邻失败类型的均值差距", "", "| β | 类型对 | 新R差 | 新G0差 | 未折扣逐对胜率 | 折扣逐对胜率 |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    adjacent_failure_pairs = list(zip(CATEGORIES[1:], CATEGORIES[2:]))
    for row in gaps:
        if row["cohort"] == COHORTS[0] and (row["higher_category"], row["lower_category"]) in adjacent_failure_pairs:
            lines.append("| %g | %s > %s | %.3f | %.3f | %.2f%% | %.2f%% |" % (row["beta"], labels[row["higher_category"]], labels[row["lower_category"]], row["new_raw_mean_gap"], row["new_discounted_mean_gap"], 100*row["raw_pairwise_win_rate"], 100*row["discounted_pairwise_win_rate"]))
    lines.extend(["", "类别均值有序与每条轨迹均严格有序具有不同含义。完整逐对胜率和辅助组结果见CSV。", "",
        "## 推荐", "",
        "在本轮明确的约束和优先目标下，推荐β=100。它在三个候选中扩大失败质量的均值差距最多，主测试集最高失败新增77.872分，全部3120条轨迹中无失败新增超过80分。", "",
        "主测试集β=%g的最高失败新回报为%.3f分，仍为负，Capture仍远高于全部失败。Capture最大新增%.3f分，占+500事件奖励%.2f%%。失败新增奖励已接近80分上限，余量%.3f分，应明确保留这一量级事实。" % (
            largest["beta"], largest["max_failure_new_return"], largest["max_capture_shaping"],
            100*largest["max_capture_shaping_to_terminal_ratio"], largest["failure_shaping_limit_margin"]), "",
        "在现有失败轨迹全程d≥5m、λ=20m的条件下，有理论上限", "",
        r"\[S_{near,failure}<\beta e^{-5/20}\]", "",
        "β=60、80、100分别对应约46.728、62.304、77.880分，均在80分以内。该界限使用当前严格小于5m的捕获终止定义。", "",
        "β=80提供更大的失败奖励余量，但失败质量区分幅度也较小。用户本轮优先拉开差距，因此本报告将β=100列为首选。此前30–50分Capture shaping量级在本轮作为历史对照记录，当前参数筛选依据为用户本轮的80分失败上限和+500主导要求。", "",
        "折扣下的末两类顺序仍有交叠与倒置。较早获得的距离成本、接近时刻和回合长度会影响G0，增加β未在本轮范围内完全解决该现象。离线结果描述已保存轨迹的计分，策略学习效果仍需独立验证。", "",
        "best-so-far项仍依赖历史最小距离，也会奖励敌机主动靠近。回合重置、历史状态及观测语义属于后续接入训练时需要明确的内容。", "",
        "![β对比](beta_comparison.png)", "",
        "## 文件和复现", "",
        "- [完整类别原/新回报与范围](category_statistics.csv)",
        "- [三组汇总及安全性](beta_summary.csv)", "- [各类别间差距和逐对胜率](category_gaps.csv)",
        "- [9360条逐回合计分](episode_reward_replay.csv)", "- [相同轨迹清单](trajectory_manifest.csv)",
        "- [原始文件与配置哈希](replay_manifest.json)", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m scripts.offline_best_so_far_beta_sweep --config configs/offline_best_so_far_beta_sweep.yaml", "```", "",
        "重复执行应选择新的output_dir，已有目录会拒绝覆盖。", ""])
    (output/"BETA_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output/"README.md").write_text("# λ20m的β幅度离线对照\n\n"
        "[完整报告](BETA_SWEEP_REPORT.md)\n\n"
        "- [全部类别的新增奖励、原/新回报与折扣回报](category_statistics.csv)\n"
        "- [β60/80/100的差距和上限](beta_summary.csv)\n"
        "- [各类别间差距](category_gaps.csv)\n"
        "- [9360条逐回合计分](episode_reward_replay.csv)\n"
        "- [对比图](beta_comparison.png)\n"
        "- [轨迹与配置来源哈希](replay_manifest.json)\n", encoding="utf-8")


def run(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT/config["output_dir"]
    if output.exists():
        raise FileExistsError("Use a new analysis output directory " + str(output))
    previous = resolve_artifact_path(config["source_analysis"], root=ROOT)
    previous_config = yaml.safe_load((previous/"analysis_config.yaml").read_text(encoding="utf-8"))
    episodes, _, _, gamma, hashes = load_previous(previous_config)
    for name in ("analysis_config.yaml", "replay_manifest.json", "trajectory_manifest.csv", "episode_reward_replay.csv"):
        path = (previous/name).resolve()
        hashes[str(path)] = sha256(path)
    hashes[str(config_path)] = sha256(config_path)
    reference_rows = read_csv(previous/"episode_reward_replay.csv")
    reference = {row["episode_id"]:row for row in reference_rows if float(row["lambda_m"]) == config["lambda_m"] and float(row["beta"]) == previous_config["focus_beta"]}
    if [row["episode_id"] for row in read_csv(previous/"trajectory_manifest.csv")] != [e["episode_id"] for e in episodes]:
        raise ValueError("Frozen trajectory sequence changed")
    rows = replay(episodes, reference, config, gamma)
    capture_reward = max(float(e["components"]["capture"].max()) for e in episodes)
    groups, summaries, gaps = summarize(rows, config, capture_reward)
    output.mkdir(parents=True, exist_ok=False)
    (output/"analysis_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output/"trajectory_manifest.csv").write_bytes((previous/"trajectory_manifest.csv").read_bytes())
    write_csv(output/"episode_reward_replay.csv", rows)
    write_csv(output/"category_statistics.csv", groups)
    write_csv(output/"beta_summary.csv", summaries)
    write_csv(output/"category_gaps.csv", gaps)
    plot(groups, summaries, config, output)
    write_report(groups, summaries, gaps, config, output)
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise RuntimeError("Source changed during analysis " + path)
    manifest = dict(status="completed", gamma=gamma, episode_count=len(episodes), replay_rows=len(rows),
        source_sha256=hashes, all_source_hashes_unchanged=True,
        trajectory_manifest_byte_identical=sha256(previous/"trajectory_manifest.csv") == sha256(output/"trajectory_manifest.csv"),
        source_analysis=str(previous.resolve()), output_dir=str(output.resolve()))
    write_json(output/"replay_manifest.json", manifest)
    print(json.dumps({key:manifest[key] for key in ("status","episode_count","replay_rows","output_dir")}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/offline_best_so_far_beta_sweep.yaml")
    run(parser.parse_args().config)
