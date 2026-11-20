"""Figures and report for offline base-attacker segment-cost calibration."""

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.offline_near_reward_replay import write_csv


CATEGORIES = ("capture", "near_miss", "moderate_failure", "other_failure", "clear_failure")
LABELS = {
    "capture": "Capture", "near_miss": "Near miss 5-10 m",
    "moderate_failure": "Failure 10-30 m", "other_failure": "Failure 30-100 m",
    "clear_failure": "Failure >100 m", "stationary": "Stationary counterfactual",
}
CHINESE_LABELS = {
    "capture": "Capture", "near_miss": "5–10m近失",
    "moderate_failure": "10–30m失败", "other_failure": "30–100m失败",
    "clear_failure": ">100m失败", "stationary": "静止反事实",
}
COLORS = {
    "capture": "#238348", "near_miss": "#2879b9", "moderate_failure": "#b77a17",
    "other_failure": "#bb4e2c", "clear_failure": "#8c386b", "stationary": "#202020",
}


def _coefficient_axis(axis, coefficients):
    positive = [float(value) for value in coefficients if value > 0]
    if positive:
        axis.set_xscale("symlog", linthresh=min(positive))
    axis.set_xticks(coefficients)
    axis.set_xticklabels(["%g" % value for value in coefficients], rotation=30)
    axis.set_xlabel("Line coefficient k [points / (m s)]")
    axis.grid(alpha=0.2)


def _group_lookup(category_rows):
    return {(row["cohort"], float(row["coefficient"]), row["category"]): row
            for row in category_rows}


def _stationary_mean(category_rows, coefficient, field):
    selected = [row for row in category_rows
                if row["cohort"] == "stationary" and float(row["coefficient"]) == coefficient]
    total = sum(int(row["episodes"]) for row in selected)
    if total == 0:
        return None
    return sum(float(row[field]) * int(row["episodes"]) for row in selected) / total


def _coefficient_plots(output, category_rows, summaries, gamma):
    coefficients = [float(row["coefficient"]) for row in summaries]
    lookup = _group_lookup(category_rows)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for name in CATEGORIES:
        selected = [lookup[("test", value, name)] for value in coefficients]
        for axis, field in ((axes[0, 0], "line_cost_magnitude_mean"),
                            (axes[0, 1], "new_return_mean"),
                            (axes[1, 0], "new_discounted_return_mean")):
            axis.plot(coefficients, [float(row[field]) for row in selected],
                      marker="o", markersize=4, label=LABELS[name], color=COLORS[name])
    for axis, field in ((axes[0, 0], "line_cost_magnitude_mean"),
                        (axes[0, 1], "new_return_mean"),
                        (axes[1, 0], "new_discounted_return_mean")):
        axis.plot(coefficients, [_stationary_mean(category_rows, value, field)
                                 for value in coefficients],
                  marker="s", markersize=4, linestyle="--", color=COLORS["stationary"],
                  label=LABELS["stationary"])
    axes[0, 0].set(title="Mean accumulated segment cost", ylabel="Cost magnitude (points)")
    axes[0, 1].set(title="Mean return with segment cost", ylabel="Undiscounted return")
    axes[1, 0].set(title="Mean discounted return, gamma=%g" % gamma,
                   ylabel="Discounted return from initial state")
    axes[1, 1].plot(coefficients,
                    [100 * float(row["stationary_near_miss_outrank_fraction"])
                     for row in summaries], marker="o", color=COLORS["stationary"],
                    label="All real Near miss pairs")
    axes[1, 1].plot(coefficients,
                    [100 * float(row["stationary_closer_near_outrank_fraction"])
                     for row in summaries], marker="s", color=COLORS["near_miss"],
                    label="Real Near miss closer than stationary")
    axes[1, 1].set(title="Paired angles, stationary vs real Near miss",
                   ylabel="Stationary return strictly exceeds real (%)", ylim=(0, 100))
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        _coefficient_axis(axis, coefficients)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.legend(fontsize=8)
    figure.suptitle("Offline segment-cost calibration | unique real trajectories | raw values", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    filename = "plots/coefficient_calibration.png"
    figure.savefig(output / filename, dpi=160)
    plt.close(figure)
    return filename


def _select_representatives(episodes):
    selected = []
    primary = [episode for episode in episodes
               if episode["cohort"] == "test" and not episode["duplicate_of"]]
    for name in CATEGORIES:
        members = [episode for episode in primary if episode["category"] == name]
        if not members:
            continue
        integrals = [float(-np.sum(episode["line_unit_cost"])) for episode in members]
        median = float(np.median(integrals))
        selected.append(min(members, key=lambda episode: (
            abs(float(-np.sum(episode["line_unit_cost"])) - median), episode["episode_id"])))
    stationary = [episode for episode in episodes if episode["cohort"] == "stationary"]
    if stationary:
        selected.append(min(stationary, key=lambda episode: (
            abs(float(episode["angle_degrees"])), episode["episode_id"])))
    return selected


def _representative_plots(output, episodes, replay_rows, coefficient):
    selected = _select_representatives(episodes)
    row_lookup = {row["episode_id"]: row for row in replay_rows
                  if float(row["coefficient"]) == coefficient}
    representatives, steps = [], []
    for index, episode in enumerate(selected, 1):
        row = row_lookup[episode["episode_id"]]
        positions = episode["states"][:, :, :3]
        time = episode["time"]
        base = np.asarray(episode["base_position"], dtype=float)
        components = episode["components"]
        line = coefficient * np.asarray(episode["line_unit_cost"])
        stationary = episode["cohort"] == "stationary"
        name = "stationary" if stationary else episode["category"]
        figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        for axis, dimensions, labels in ((axes[0, 0], (0, 1), ("X (m)", "Y (m)")),
                                         (axes[0, 1], (0, 2), ("X (m)", "Z (m)"))):
            horizontal, vertical = dimensions
            axis.plot(positions[:, 0, horizontal], positions[:, 0, vertical],
                      color="tab:blue", label="P0")
            axis.plot(positions[:, 1, horizontal], positions[:, 1, vertical],
                      color="tab:red", label="Direct Attacker")
            axis.plot([base[horizontal], positions[0, 1, horizontal]],
                      [base[vertical], positions[0, 1, vertical]],
                      color="gray", linestyle=":", label="Initial segment")
            axis.plot([base[horizontal], positions[-1, 1, horizontal]],
                      [base[vertical], positions[-1, 1, vertical]],
                      color="black", linestyle="--", linewidth=1, label="Terminal segment")
            for agent, color in ((0, "tab:blue"), (1, "tab:red")):
                axis.scatter(positions[0, agent, horizontal], positions[0, agent, vertical],
                             marker="^", color=color, s=45, zorder=4)
                axis.scatter(positions[-1, agent, horizontal], positions[-1, agent, vertical],
                             marker="o", color=color, s=35, zorder=4)
            axis.scatter(base[horizontal], base[vertical], marker="*", color="black",
                         s=110, label="Base", zorder=5)
            axis.set_aspect("equal", adjustable="datalim")
            axis.set(xlabel=labels[0], ylabel=labels[1], title="Equal physical axis scale")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        axes[1, 0].plot(time, episode["distances"], color="gray", label="P0-attacker distance")
        axes[1, 0].plot(time, episode["segment_distances"], color=COLORS[name],
                        label="3D point-to-segment distance")
        axes[1, 0].set(xlabel="Time (s)", ylabel="Distance (m)", title="Recorded geometry")
        axes[1, 0].grid(alpha=0.2)
        axes[1, 0].legend(fontsize=8)
        for field, label in (("original", "Original four-term reward"),
                             ("near_shaping", "Best-so-far shaping")):
            axes[1, 1].plot(time[1:], np.cumsum(components[field]), label=label)
        axes[1, 1].plot(time[1:], np.cumsum(line), color="tab:purple", label="Added segment cost")
        axes[1, 1].plot(time[1:], np.cumsum(components["total"] + line),
                        color="black", linestyle="--", label="New cumulative total")
        axes[1, 1].set(xlabel="Time (s)", ylabel="Cumulative reward (points)",
                       title="Raw cumulative components, k=%g" % coefficient)
        axes[1, 1].grid(alpha=0.2)
        axes[1, 1].legend(fontsize=8)
        origin = "OFFLINE STATIONARY COUNTERFACTUAL" if stationary else "SAVED REAL MODEL TRAJECTORY"
        identity = ("scene %s" % episode["scenario_id"] if stationary
                    else "seed %s" % episode["training_seed"])
        figure.suptitle("%s | %s | %s | %+.3f deg\nMin %.3f m | duration %.1f s | line cost %.3f | R %.3f -> %.3f" % (
            origin, LABELS[name], identity, float(episode["angle_degrees"]),
            float(np.min(episode["distances"])), float(time[-1]), float(line.sum()),
            float(row["baseline_return"]), float(row["new_return"])), fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        filename = "plots/representative_%02d_%s.png" % (index, name)
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)
        representatives.append(dict(
            episode_id=episode["episode_id"], cohort=episode["cohort"], category=name,
            training_seed=episode["training_seed"], model=episode["model"],
            checkpoint_step=episode["checkpoint_step"], angle_degrees=episode["angle_degrees"],
            min_distance=float(np.min(episode["distances"])), event_time=float(time[-1]),
            coefficient=coefficient, baseline_return=row["baseline_return"],
            new_return=row["new_return"], line_penalty=row["line_penalty"],
            trajectory=episode["trajectory"], component_plot=filename))
        for step in range(len(time) - 1):
            steps.append(dict(
                episode_id=episode["episode_id"], cohort=episode["cohort"], step=step,
                time_start=float(time[step]), time_end=float(time[step + 1]),
                pursuer_attacker_distance_before=float(episode["distances"][step]),
                pursuer_attacker_distance_after=float(episode["distances"][step + 1]),
                segment_distance_before=float(episode["segment_distances"][step]),
                segment_distance_after=float(episode["segment_distances"][step + 1]),
                distance_cost=float(components["distance_cost"][step]),
                survival=float(components["survival"][step]),
                capture=float(components["capture"][step]),
                defense_failure=float(components["defense_failure"][step]),
                original_reward=float(components["original"][step]),
                near_shaping=float(components["near_shaping"][step]),
                baseline_reward=float(components["total"][step]),
                coefficient=coefficient, line_cost=float(line[step]),
                new_reward=float(components["total"][step] + line[step])))
    write_csv(output / "representatives.csv", representatives)
    write_csv(output / "representative_step_components.csv", steps)
    return representatives


def _fmt(value, digits=3):
    return "—" if value is None else ("%.*f" % (digits, float(value)))


def _write_report(output, episodes, replay_rows, summaries, category_rows,
                  recommendation, manifest, config, overview, representatives, coefficient):
    coefficients = [float(row["coefficient"]) for row in summaries]
    gamma = float(manifest["gamma"])
    primary = [episode for episode in episodes
               if episode["cohort"] == "test" and not episode["duplicate_of"]]
    counts = Counter(episode["cohort"] for episode in episodes)
    unique = Counter(episode["cohort"] for episode in episodes if not episode["duplicate_of"])
    labels = CHINESE_LABELS
    lookup = _group_lookup(category_rows)
    recommended = recommendation.get("recommended_coefficient")
    lines = ["# 敌机—基地线段偏离成本离线标定", "", "## 结论", ""]
    if recommended is None:
        lines.extend(["本轮暂定筛查下没有推荐项。已扫描的正系数未同时满足本轮量级偏好与静止对照检查，后续可继续研究几何目标及其量级。",
                      "静止对照是本轮标定的重要反例，增加线段偏离成本提高了它相对于部分实际Near miss轨迹的回报排序。", ""])
    else:
        lines.extend(["本批离线检查得到候选k=%g，单位为分/(m·s)。该候选对应已保存轨迹的重新计分，训练收益有待独立验证。" % recommended, ""])
    lines.extend(["## 数据来源与去重", "",
        "主数据读取本轮B组三个Seed的Best与Final保存轨迹。固定0°逐Checkpoint历史轨迹单独统计。静止对照复用同角度Direct Attacker轨迹，令我方保持配置初始位置与零速度，并重新截断到首个捕获、失守或超时事件。", "",
        "| 数据组 | 文件记录数 | 去重后回合数 |", "| --- | ---: | ---: |"])
    for cohort, label in (("test", "真实模型角度测试"), ("history", "真实固定0°历史"),
                          ("stationary", "离线静止反事实")):
        lines.append("| %s | %d | %d |" % (label, counts[cohort], unique[cohort]))
    lines.extend(["", "主表按去重后的真实测试回合计算。相同场景与物理轨迹的重复记录保留来源映射，重复项不增加主表权重。", "",
        "静止反事实与真实模型记录分别汇总。它使用保存的独立敌机运动作为输入，其终止和奖励经过重新计算。静止在本任务中是可执行控制，静止失败本身也可能属于Near miss。", "",
        "| 真实测试类别 | 去重后回合数 |", "| --- | ---: |"])
    for name in CATEGORIES:
        lines.append("| %s | %d |" % (labels[name], sum(episode["category"] == name for episode in primary)))
    lines.extend(["", "Capture沿用敌我三维距离严格小于5m。Near miss对应失败且5≤最小距离≤10m，随后依次为10–30m、30–100m与大于100m。", "",
        "## 公式与扫描设置", "",
        r"\[s_t=\operatorname{clip}\!\left(\frac{(p_t-b)\cdot(e_t-b)}{\|e_t-b\|^2},0,1\right),\quad q_t=b+s_t(e_t-b),\quad d_{\perp,t}=\|p_t-q_t\|\]", "",
        r"\[r_{line,t}=-k\,d_{\perp,t+1}\,(t_{t+1}-t_t),\qquad r'_t=r_{B,t}+r_{line,t}\]", "",
        "p、e、b分别为我方、敌机和基地三维位置。敌机与基地重合时，线段退化为基地点。每步使用真实积分后位置和保存时间间隔，末步计入。r_B包含源B组原四项奖励和历史最小距离shaping。", "",
        "有限线段距离包含线段内部的垂直偏离与投影越出端点后的纵向距离。敌机飞过我方之后，线段随敌机收缩，最近相遇后的扣分仍会继续累积。分解量保存在逐回合统计中。", "",
        "扫描k为%s，单位分/(m·s)。k=0提供原B组回报基线。折扣回报使用源训练γ=%g。" % (
            "、".join("%g" % value for value in coefficients), gamma), "",
        "扫描网格与以下候选检查阈值由本轮分析选定，均保存在独立analysis_config.yaml中。", "",
        "~~~json", json.dumps(config["calibration_checks"], ensure_ascii=False, indent=2), "~~~", "",
        "成本阈值约束成功回合受罚程度及明显失败的区分量级。静止检查限制静止反事实超过同角度真实Near miss的比例相对k=0增加幅度。这些阈值体现本轮保守量级偏好，推荐集合随筛查偏好变化。", "",
        "## 系数与累计惩罚量级", "", "以下为各类平均扣分的绝对值。", "",
        "| k | Capture | Near miss | 10–30m失败 | 30–100m失败 | >100m失败 | 静止反事实 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in summaries:
        values = [row[name + "_mean_cost"] for name in CATEGORIES]
        lines.append("| %g | %s | %s | %s | %s | %s | %s |" % (
            row["coefficient"], *[_fmt(value) for value in values], _fmt(row["stationary_mean_cost"])))
    for field, title in (("new_return_mean", "未折扣新回报"),
                          ("new_discounted_return_mean", "γ=%g折扣新回报" % gamma)):
        lines.extend(["", "## 各类别平均%s" % title, "",
            "| k | Capture | Near miss | 10–30m失败 | 30–100m失败 | >100m失败 | 静止反事实 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for value in coefficients:
            values = [lookup[("test", value, name)][field] for name in CATEGORIES]
            lines.append("| %g | %s | %s | %s | %s | %s | %s |" % (
                value, *[_fmt(number) for number in values],
                _fmt(_stationary_mean(category_rows, value, field))))
    lines.extend(["", "## 静止对照与排序风险", "",
        "基地是参考线段端点，我方初态天然靠近线段。静止对照通常只承担较小偏离成本，实际前出过程中产生偏差的Near miss会被扣除更多分。", "",
        "静止轨迹与实际Near miss之间的原始最小距离也有差异，主动飞行本身不构成应得更高回报的充分条件。为进一步检查副作用，下表还单独统计实际最小距离比同角度静止更近的子集。", "",
        "| k | 静止平均新R | 静止超过全部配对Near miss | 相对k=0增加/百分点 | 静止超过更近Near miss | 暂定候选 |",
        "| ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in summaries:
        lines.append("| %g | %s | %.2f%% | %+.2f | %.2f%% | %s |" % (
            row["coefficient"], _fmt(row["stationary_mean_new_return"]),
            100 * float(row["stationary_near_miss_outrank_fraction"]),
            100 * float(row["stationary_near_miss_outrank_increase"]),
            100 * float(row["stationary_closer_near_outrank_fraction"]),
            "是" if row["calibration_candidate"] else "否"))
    closer_count = int(summaries[0]["closer_near_miss_count"])
    lines.extend(["", "实际Near miss比同角度静止更近的配对样本共有%d条。该子集用于观察新增项是否压过已获得的接近质量优势。完整回报还受到实际路径、接近时刻及回合长度影响。" % closer_count, "",
        "![系数、回报和静止对照](%s)" % overview, "",
        "## 成本发生位置与时段", "",
        "下表按几何时间积分比较成本来源。基地后侧和敌机外侧按投影位置分组，最近相遇后按每条真实轨迹的最小敌我距离时刻分组，时段与空间分组可以重叠。", "",
        "| 类别 | 线段积分/m·s | 无限直线积分/m·s | 基地后侧积分/m·s | 敌机外侧积分/m·s | 最近相遇后占比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name in CATEGORIES:
        group = lookup[("test", coefficients[0], name)]
        integral = float(group["segment_integral_mean"])
        after_fraction = float(group["after_closest_integral_mean"]) / integral if integral > 0 else 0.0
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.2f%% |" % (
            labels[name], integral, float(group["infinite_line_integral_mean"]),
            float(group["before_base_integral_mean"]), float(group["beyond_attacker_integral_mean"]),
            100 * after_fraction))
    lines.extend(["", "有限线段的端点限制增加了越出端点时的成本。该项同时评价横向、高度与线段纵向位置，持续积分也会放大错过目标之后的偏离。", "",
        "## 捕获奖励主导关系", "",
        "下表直接检查主真实测试集中的成功与失败回报极值。差值为最低成功回报减最高失败回报。", "",
        "| k | 最低成功新R | 最高失败新R | R极值间隔 | G极值间隔 |",
        "| ---: | ---: | ---: | ---: | ---: |"])
    for value in coefficients:
        members = [row for row in replay_rows if row["cohort"] == "test"
                   and not row["duplicate_of"] and float(row["coefficient"]) == value]
        captures = [row for row in members if row["category"] == "capture"]
        failures = [row for row in members if row["category"] != "capture"]
        min_capture = min(float(row["new_return"]) for row in captures)
        max_failure = max(float(row["new_return"]) for row in failures)
        discounted_gap = min(float(row["new_discounted_return"]) for row in captures) - max(
            float(row["new_discounted_return"]) for row in failures)
        lines.append("| %g | %.3f | %.3f | %.3f | %.3f |" % (
            value, min_capture, max_failure, min_capture - max_failure, discounted_gap))
    lines.extend(["", "相邻类别的全部轨迹对胜率见pairwise_rankings.csv，逐回合完整结果见episode_reward_replay.csv。", "",
        "## 代表轨迹与逐时间分量", "",
        "各真实类别选择线段距离时间积分最接近该类中位数的已去重轨迹。静止样本选择来袭角度绝对值最小者。图中三角形为起点、圆点为终点，XY与XZ均采用米制等比例。", "",
        "代表图统一使用k=%g作为%s。全部系数的统计保留在CSV中。" % (
            coefficient, "推荐候选示例" if recommended is not None else "诊断示例"), "",
        "| 类型 | Seed | 模型 | 角度/° | min/m | 原B回报 | 加线段成本后 | 图 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"])
    for row in representatives:
        lines.append("| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | [打开](%s) |" % (
            labels[row["category"]], row["training_seed"], row["model"], row["angle_degrees"],
            row["min_distance"], row["baseline_return"], row["new_return"], row["component_plot"]))
    for name in ("near_miss", "stationary"):
        source = next((row for row in representatives if row["category"] == name), None)
        if source:
            lines.extend(["", "![%s的几何和累计分项](%s)" % (labels[name], source["component_plot"])])
    lines.extend(["", "代表样本的全部原四项、best-so-far与line逐步值见representative_step_components.csv。图中保留原始数据，无平滑。", "",
        "## 推荐结果与解释范围", "",
        "~~~json", json.dumps(recommendation, ensure_ascii=False, indent=2), "~~~", "",
        "线段成本能够区分空间偏离，其对前出过程和相遇后阶段的持续扣分也会改变静止与主动接近的相对价值。样本回报排序需要与静止反例同时判断。", "",
        "本轮结果描述冻结轨迹的重新计分。轨迹、终止时刻、角度和控制动作来自既有记录，静止反事实另行标注。未来训练对分布、探索和策略保持的影响有待独立实验。", "",
        "原先500角度测试场景本轮已用于奖励开发标定，后续独立性能检验宜使用新场景集。固定0°历史图用于补充路径变化，与主随机角度统计分别汇总。", "",
        "## 数值验证与复现", "",
        "专项测试覆盖线段几何、时间积分、奖励重建与反事实终止。实际测试命令、通过数量和耗时见verification.json。", "",
        "源B组逐步奖励复算的最大绝对误差为%s。源文件SHA256与末次核验结果保存在replay_manifest.json和verification.json。" % (
            "%.3e" % manifest["max_reward_reconstruction_error"]), "",
        "~~~powershell", ".\\.venv\\Scripts\\python.exe -m scripts.offline_line_cost_replay --config configs/offline_line_cost_replay.yaml", "~~~", "",
        "重放入口为每次运行创建独立输出目录，已有输出通过目录保护保留。", "",
        "[参数与检查摘要](coefficient_summary.csv) · [逐类别统计](category_statistics.csv) · [逐回合结果](episode_reward_replay.csv)", "",
        "[相邻类别胜率](pairwise_rankings.csv) · [代表轨迹](representatives.csv) · [逐时间分项](representative_step_components.csv)", "",
        "[配置快照](analysis_config.yaml) · [来源清单与哈希](replay_manifest.json)", ""])
    report = output / "LINE_COST_CALIBRATION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    navigation = ["# 线段偏离成本离线标定", "",
        "[完整报告](LINE_COST_CALIBRATION_REPORT.md)", "",
        "- [系数、回报与静止对照图](%s)" % overview,
        "- [系数汇总](coefficient_summary.csv)",
        "- [逐类别统计](category_statistics.csv)",
        "- [全部逐回合重放](episode_reward_replay.csv)",
        "- [代表轨迹与分量图](representatives.csv)",
        "- [逐时间奖励分项](representative_step_components.csv)",
        "- [来源与哈希](replay_manifest.json)", ""]
    (output / "README.md").write_text("\n".join(navigation), encoding="utf-8")
    return report.name


def write_results(output, episodes, replay_rows, summaries, category_rows, pairwise_rows,
                  recommendation, manifest, config):
    """Write figures, representative raw samples and the Chinese report."""
    output = Path(output)
    (output / "plots").mkdir()
    gamma = float(manifest["gamma"])
    recommended = recommendation.get("recommended_coefficient")
    coefficient = float(config["diagnostic_coefficient"] if recommended is None else recommended)
    if coefficient not in [float(value) for value in config["coefficients"]]:
        raise ValueError("Representative coefficient must be present in the scanned grid")
    overview = _coefficient_plots(output, category_rows, summaries, gamma)
    representatives = _representative_plots(output, episodes, replay_rows, coefficient)
    report = _write_report(output, episodes, replay_rows, summaries, category_rows,
                           recommendation, manifest, config, overview, representatives, coefficient)
    return dict(report=report, overview=overview,
                representatives=[row["component_plot"] for row in representatives],
                diagnostic_coefficient=coefficient)
