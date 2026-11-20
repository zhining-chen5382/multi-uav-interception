"""Figures and Chinese report for frozen-position near-reward replay."""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = ("capture", "near_miss", "moderate_failure", "other_failure", "clear_failure")
LABELS = {
    "capture": "Capture", "near_miss": "Near miss 5-10 m",
    "moderate_failure": "Failure 10-30 m", "other_failure": "Failure 30-100 m",
    "clear_failure": "Failure >100 m",
}
CHINESE_LABELS = {
    "capture": "Capture", "near_miss": "5–10m近失",
    "moderate_failure": "10–30m失败", "other_failure": "30–100m失败",
    "clear_failure": ">100m失败",
}
DATASETS = {"reference_B": "原B组固定样本", "current_k0p1": "本轮k=0.1样本"}


def _number(value, digits=3):
    return "—" if value is None else "%.*f" % (digits, float(value))


def _overview(output, categories):
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    titles = {"reference_B": "Reference B trajectories", "current_k0p1": "Current k=0.1 trajectories"}
    for column, dataset in enumerate(DATASETS):
        lookup = {row["category"]: row for row in categories
                  if row["dataset"] == dataset and row["cohort"] == "test"}
        names = [name for name in CATEGORIES if name in lookup]
        selected = [lookup[name] for name in names]
        indices = np.arange(len(names))
        labels = ["%s\nn=%d" % (LABELS[name], lookup[name]["episodes"]) for name in names]
        comparisons = (
            ("old_near_return_mean", "Original near", "active_near_return_mean", "Corrected near"),
            ("active_near_return_mean", "Corrected near", "line_return_mean", "Line cost magnitude"),
        )
        for row_index, (left, left_label, right, right_label) in enumerate(comparisons):
            axis = axes[row_index, column]
            left_values = [float(row[left]) for row in selected]
            right_values = [abs(float(row[right])) for row in selected]
            axis.bar(indices - 0.18, left_values, width=0.36, color="#6a9ac4", label=left_label)
            axis.bar(indices + 0.18, right_values, width=0.36,
                     color="#28764d" if row_index == 0 else "#b56b40", label=right_label)
            axis.set_xticks(indices)
            axis.set_xticklabels(labels, fontsize=8)
            axis.set(title=titles[dataset], ylabel="Mean episode component (points)")
            axis.grid(axis="y", alpha=0.2)
            axis.legend(fontsize=8)
    figure.suptitle("Offline near-reward attribution | unique saved episodes | raw class means")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    filename = "plots/old_vs_active_near.png"
    figure.savefig(output / filename, dpi=160)
    plt.close(figure)
    return filename


def _representative_plots(output, representatives):
    results = []
    for index, example in enumerate(representatives):
        row = example["row"]
        time = np.asarray(example["time"])
        figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(time, example["distances"], color="#777777", label="Actual distance")
        axes[0].plot(time, example["best_distances"], color="#246eab", label="Actual historical minimum")
        axes[0].plot(time[1:], example["counterfactual_distances"], color="#bd8b31",
                     linestyle="--", label="Next attacker, frozen previous pursuer")
        axes[0].set(ylabel="3D distance (m)", title="Recorded and one-step counterfactual distances")
        axes[1].plot(time[1:], example["original_near"], label="Original near", color="#6a9ac4")
        axes[1].plot(time[1:], example["active_near"], label="Corrected near", color="#28764d")
        axes[1].plot(time[1:], example["removed_near"], label="Removed near", color="#bd8b31", alpha=0.8)
        axes[1].set(ylabel="Reward per transition", title="Raw per-step near components")
        for field, label, color in (("original_near", "Original near", "#6a9ac4"),
                                    ("active_near", "Corrected near", "#28764d"),
                                    ("line", "Line cost", "#b56b40")):
            axes[2].plot(time, np.r_[0.0, np.cumsum(example[field])], label=label, color=color)
        axes[2].set(xlabel="Time (s)", ylabel="Cumulative component (points)",
                    title="Undiscounted component accumulation")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        origin = "STATIONARY COUNTERFACTUAL" if row["cohort"] == "stationary" else "SAVED REAL TRAJECTORY"
        figure.suptitle("%s | %s | seed %s | %s | %s\nMin %.3f m | old near %.3f | corrected %.3f | removed %.3f" % (
            row["dataset"], origin, row["training_seed"], row["model"], LABELS[row["category"]],
            row["min_distance"], row["old_near_return"], row["active_near_return"], row["removed_near_return"]), fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        filename = "plots/representative_%02d_%s.png" % (index, row["category"])
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)
        results.append(dict(episode_id=example["episode_id"], dataset=row["dataset"],
                            cohort=row["cohort"], category=row["category"], plot=filename))
    return results


def _return_tables(lines, groups, gamma):
    for suffix, title in (("", "未折扣回报"), ("_discounted", "γ=%g折扣回报" % gamma)):
        lines.extend(["", "### " + title, "",
            "| 类别 | 数量 | 原near，无line | 校正near，无line | 原near，含line | 校正near，含line |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"])
        fields = ("old_without_line_return", "new_without_line_return", "old_with_line_return", "new_with_line_return")
        for name in CATEGORIES:
            if name not in groups:
                continue
            row = groups[name]
            values = [_number(row[field + suffix + "_mean"]) for field in fields]
            lines.append("| %s | %d | %s |" % (CHINESE_LABELS[name], row["episodes"], " | ".join(values)))
        for field in ("new_without_line_return", "new_with_line_return"):
            ordered = sorted(groups, key=lambda name: float(groups[name][field + suffix + "_mean"]), reverse=True)
            label = "校正near，无line" if field == "new_without_line_return" else "校正near，含line"
            lines.extend(["", "%s的现有类别均值从高到低为 %s。" % (
                label, " > ".join(CHINESE_LABELS[name] for name in ordered))])
    absent = [CHINESE_LABELS[name] for name in CATEGORIES if name not in groups]
    if absent:
        lines.extend(["", "本数据组缺少%s样本，相关排序仍需后续样本验证。" % "、".join(absent)])


def _report(output, rows, categories, pairwise, stationary, plots, overview, manifest):
    unique = [row for row in rows if not row["duplicate_of"]]
    gamma, beta = float(manifest["gamma"]), float(manifest["beta"])
    scale, coefficient = float(manifest["length_scale_m"]), float(manifest["line_coefficient"])
    lines = ["# 去除敌机被动接近贡献的离线奖励重放", "", "## 方法与范围", "",
        "本轮在既有完整轨迹上重新计算near分项。真实动作、轨迹和终止时刻用于全部配对计算。静止反事实单独标注，采用保存的独立Direct Attacker运动。", "",
        r"\[d^{cf}_{t+1}=\|a_{t+1}-p_t\|,\quad b^{cf}_{t+1}=\min(b_t,d^{cf}_{t+1}),\quad b_{t+1}=\min(b_t,\|a_{t+1}-p_{t+1}\|)\]", "",
        r"\[r^{active}_{near,t}=\beta\max(0,e^{-b_{t+1}/\lambda}-e^{-b^{cf}_{t+1}/\lambda})\]", "",
        "β=%g，λ=%gm，折扣γ=%g。line采用k=%g分/(m·s)，使用积分后闭线段距离乘实际dt。" % (beta, scale, gamma, coefficient), "",
        "该反事实把敌机固定在实际下一位置，衡量我方本步实际位移的边际贡献，惯性滑行也计入。历史最小距离始终沿用真实轨迹，正截断使未突破纪录或未优于冻结位置反事实时奖励为零。", "",
        "逐步满足0≤校正near≤原near，因此整回合校正near保留原奖励的上界。已有距离区间在绕回后不能再次获得奖励。", "",
        "## 样本与去重", "", "| 数据组 | 回合类型 | 文件记录数 | 去重后回合数 |", "| --- | --- | ---: | ---: |"]
    counts = Counter((row["dataset"], row["cohort"]) for row in rows)
    unique_counts = Counter((row["dataset"], row["cohort"]) for row in unique)
    for key, count in counts.items():
        lines.append("| %s | %s | %d | %d |" % (DATASETS.get(key[0], key[0]), key[1], count, unique_counts[key]))
    lines.extend(["", "类别沿用来源清单。主随机角度测试、固定0°历史和静止反事实分别汇总，重复物理轨迹保留来源映射。", "",
                  "![Near分项与线段成本](%s)" % overview])
    for dataset, label in DATASETS.items():
        groups = {row["category"]: row for row in categories
                  if row["dataset"] == dataset and row["cohort"] == "test"}
        if not groups:
            continue
        lines.extend(["", "## " + label, "", "| 类别 | 数量 | 原near | 校正near | 去除贡献 | line成本 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for name in CATEGORIES:
            if name in groups:
                row = groups[name]
                fields = ("old_near_return", "active_near_return", "removed_near_return", "line_return")
                lines.append("| %s | %d | %s |" % (CHINESE_LABELS[name], row["episodes"],
                    " | ".join(_number(row[field + "_mean"]) for field in fields)))
        _return_tables(lines, groups, gamma)
    lines.extend(["", "## 静止与真实近失的配对比较", "",
        "静止超过率按相同来袭场景严格比较回报。更近近失子集要求真实最小距离小于同场景静止距离。", "",
        "| 数据组 | 回报字段 | 静止平均回报 | 近失数 | 静止超过近失 | 更近近失数 | 静止超过更近近失 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in stationary:
        fraction = row.get("stationary_outrank_fraction")
        closer_fraction = row.get("stationary_outrank_closer_fraction")
        lines.append("| %s | %s | %s | %d | %s | %d | %s |" % (
            DATASETS.get(row["dataset"], row["dataset"]), row["return_field"],
            _number(row["stationary_mean_return"]), row["near_count"],
            "—" if fraction is None else "%.2f%%" % (100 * fraction), row["closer_near_count"],
            "—" if closer_fraction is None else "%.2f%%" % (100 * closer_fraction)))
    static = [row for row in unique if row["cohort"] == "stationary"]
    max_static = max((abs(float(row["active_near_return"])) for row in static), default=0.0)
    lines.extend(["", "全部静止反事实的校正near累计绝对值最大为%s分。" % _number(max_static, 12), "",
        "## 代表轨迹", "", "代表样本选择与逐步数据见representatives.csv和representative_step_components.csv。图中保留原始逐步值与原始累计值。", "",
        "| 数据组 | 回合类型 | 类别 | 轨迹ID | 图 |", "| --- | --- | --- | --- | --- |"])
    for row in plots:
        lines.append("| %s | %s | %s | %s | [打开](%s) |" % (
            DATASETS.get(row["dataset"], row["dataset"]), row["cohort"], CHINESE_LABELS[row["category"]],
            row["episode_id"], row["plot"]))
    for cohort, name in (("test", "near_miss"), ("stationary", "near_miss")):
        selected = next((row for row in plots if row["cohort"] == cohort and row["category"] == name), None)
        if selected:
            lines.extend(["", "![%s](%s)" % ("实际近失" if cohort == "test" else "静止反事实", selected["plot"])])
    lines.extend(["", "## 结果解释与后续验证", "",
        "校正near描述我方位移相对于冻结位置的奖励贡献。被去除部分依赖这个逐步反事实定义，不能直接解释为真实物理接近距离的固定比例。", "",
        "去除被动收益使near正奖励总量下降，固定k下line的相对权重会上升。远距离指数项仍然很弱，是否能够促进主动前出需要进一步独立训练验证。", "",
        "原距离成本仍随双方真实位置变化，基地端点的低line成本也继续影响总回报。静止near归零后，应继续根据上表检查整体回报是否偏向静止。", "",
        "500角度场景已经参与奖励开发，本轮用于同轨迹诊断比较。固定0°历史轨迹提供补充证据，后续独立性能结论需要新的测试场景。", "",
        "各相邻类别的完整轨迹对胜率见pairwise_rankings.csv。类别平均排序与每条轨迹的排序分别保留。", "",
        "## 数值与来源", "",
        "源奖励重建最大逐步误差为%s。完整来源哈希、数学不变量检查与样本映射见replay_manifest.json和verification.json。" % (
            "%.3e" % float(manifest["max_reward_reconstruction_error"])), "",
        "[逐回合结果](episode_reward_replay.csv) · [逐类别统计](category_statistics.csv) · [相邻类别胜率](pairwise_rankings.csv)", "",
        "[配置快照](analysis_config.yaml) · [样本清单](trajectory_manifest.csv) · [来源与哈希](replay_manifest.json)", ""])
    filename = "ACTIVE_NEAR_REPLAY_REPORT.md"
    (output / filename).write_text("\n".join(lines), encoding="utf-8")
    return filename


def write_results(output, rows, category_rows, pairwise_rows, stationary_rows,
                  representatives, manifest, config):
    """Render raw replay summaries without loading policies or environments."""
    output = Path(output)
    (output / "plots").mkdir()
    overview = _overview(output, category_rows)
    plots = _representative_plots(output, representatives)
    report = _report(output, rows, category_rows, pairwise_rows, stationary_rows, plots, overview, manifest)
    navigation = ["# 去除被动接近贡献的离线重放", "", "[完整报告](%s)" % report, "",
        "- [Near与line分项对比](%s)" % overview,
        "- [逐类别统计](category_statistics.csv)", "- [逐回合回报](episode_reward_replay.csv)",
        "- [静止配对统计](stationary_comparison.csv)", "- [代表轨迹](representatives.csv)",
        "- [逐时间奖励分项](representative_step_components.csv)", "- [来源与哈希](replay_manifest.json)", ""]
    (output / "README.md").write_text("\n".join(navigation), encoding="utf-8")
    return dict(report=report, overview=overview, representatives=plots)
