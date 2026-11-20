"""Plots and Chinese report for offline active-near and line-cost calibration."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = {
    "reference_B": ("原B组固定样本", "Reference B trajectories"),
    "current_k0p1": ("当前k=0.1样本", "Current k=0.1 trajectories"),
}
CATEGORIES = ("capture", "near_miss", "moderate_failure", "other_failure", "clear_failure")
CATEGORY_LABELS = {
    "capture": "Capture", "near_miss": "Near miss 5-10 m",
    "moderate_failure": "Failure 10-30 m", "other_failure": "Failure 30-100 m",
    "clear_failure": "Failure >100 m",
}
CHINESE_CATEGORIES = {
    "capture": "Capture", "near_miss": "5–10m近失", "moderate_failure": "10–30m失败",
    "other_failure": "30–100m失败", "clear_failure": ">100m失败",
}


def _number(value, digits=3):
    if value is None or not np.isfinite(float(value)):
        return "—"
    return "%.*f" % (digits, float(value))


def _percent(value):
    return "—" if value is None else _number(100 * float(value), 2) + "%"


def _parameters_match(row, beta, coefficient):
    return (np.isclose(float(row["beta"]), float(beta), rtol=0, atol=1e-12)
            and np.isclose(float(row["k"]), float(coefficient), rtol=0, atol=1e-12))


def _grid_values(rows, dataset, field, betas, coefficients, multiplier=1.0):
    grid = np.full((len(betas), len(coefficients)), np.nan)
    selected = [row for row in rows if row["dataset"] == dataset and row["cohort"] == "test"]
    for row_index, beta in enumerate(betas):
        for column_index, coefficient in enumerate(coefficients):
            source = next((row for row in selected if _parameters_match(row, beta, coefficient)), None)
            if source is not None and source.get(field) is not None:
                grid[row_index, column_index] = float(source[field]) * multiplier
    return grid


def _heatmaps(output, summaries, manifest):
    betas = list(manifest.get("beta_grid", []))
    coefficients = list(manifest.get("k_grid", []))
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    fields = ("stationary_outrank_closer_fraction", "near_moderate_discounted_gap")
    grids = {(index, dataset): _grid_values(summaries, dataset, field, betas, coefficients,
                                           100.0 if index == 0 else 1.0)
             for index, field in enumerate(fields) for dataset in DATASETS}
    finite_gaps = np.concatenate([grid[np.isfinite(grid)] for (index, _), grid in grids.items()
                                  if index == 1])
    gap_limit = max(float(np.max(np.abs(finite_gaps))) if finite_gaps.size else 1.0, 1e-6)
    for column, (dataset, (_, title)) in enumerate(DATASETS.items()):
        for row_index in range(2):
            axis = axes[row_index, column]
            grid = grids[row_index, dataset]
            if not grid.size or not np.isfinite(grid).any():
                axis.text(0.5, 0.5, "No eligible paired observations", ha="center", va="center",
                          transform=axis.transAxes)
                axis.set_axis_off()
                continue
            image = axis.imshow(grid, aspect="auto", origin="lower",
                                cmap="YlOrRd" if row_index == 0 else "RdYlGn",
                                vmin=0.0 if row_index == 0 else -gap_limit,
                                vmax=100.0 if row_index == 0 else gap_limit)
            axis.set_xticks(np.arange(len(coefficients)))
            axis.set_xticklabels(["%g" % value for value in coefficients])
            axis.set_yticks(np.arange(len(betas)))
            axis.set_yticklabels(["%g" % value for value in betas])
            metric = ("Stationary beats closer near miss in R (%)" if row_index == 0
                      else "Near miss minus 10-30 m failure in discounted G")
            axis.set(xlabel="Line coefficient k", ylabel="Active-near beta", title=title + "\n" + metric)
            for index in np.ndindex(grid.shape):
                value = grid[index]
                label = "—" if not np.isfinite(value) else "%.*f" % (1 if row_index == 0 else 3, value)
                axis.text(index[1], index[0], label, ha="center", va="center", fontsize=9,
                          color="white" if row_index == 0 and value > 60 else "black")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Frozen-trajectory calibration | raw paired statistics | lower top and higher bottom preferred")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    filename = "plots/active_near_line_grid.png"
    figure.savefig(output / filename, dpi=170)
    plt.close(figure)
    return filename


def _candidate_plot(output, categories, recommendation, manifest):
    beta, coefficient = recommendation.get("recommended_beta"), recommendation.get("recommended_k")
    if beta is None or coefficient is None:
        return None
    baseline_beta, baseline_k = manifest["source_beta"], manifest["source_k"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for column, (dataset, (_, title)) in enumerate(DATASETS.items()):
        members = [row for row in categories if row["dataset"] == dataset and row["cohort"] == "test"]
        baseline = {row["category"]: row for row in members if _parameters_match(row, baseline_beta, baseline_k)}
        candidate = {row["category"]: row for row in members if _parameters_match(row, beta, coefficient)}
        names = [name for name in CATEGORIES if name in baseline and name in candidate]
        indices = np.arange(len(names))
        for row_index, field in enumerate(("total_return_mean", "discounted_return_mean")):
            axis = axes[row_index, column]
            if not names:
                axis.text(0.5, 0.5, "No matching class data", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            axis.bar(indices - 0.18, [baseline[name][field] for name in names], width=0.36,
                     label="Baseline beta=%g, k=%g" % (baseline_beta, baseline_k), color="#789cbb")
            axis.bar(indices + 0.18, [candidate[name][field] for name in names], width=0.36,
                     label="Candidate beta=%g, k=%g" % (beta, coefficient), color="#28764d")
            axis.set_xticks(indices)
            axis.set_xticklabels([CATEGORY_LABELS[name] for name in names], fontsize=8)
            axis.set(title=title, ylabel="Mean episode return R" if row_index == 0 else "Mean discounted return G")
            axis.grid(axis="y", alpha=0.2)
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.legend(fontsize=8)
    figure.suptitle("Candidate and corrected-near baseline on identical saved trajectories")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    filename = "plots/candidate_vs_baseline_returns.png"
    figure.savefig(output / filename, dpi=170)
    plt.close(figure)
    return filename


def _candidate_tables(lines, categories, recommendation, title="推荐组合的类别回报"):
    beta, coefficient = recommendation.get("recommended_beta"), recommendation.get("recommended_k")
    if beta is None or coefficient is None:
        return
    lines.extend(["", "## " + title, "", "本节采用β=%g、k=%g。" % (beta, coefficient), ""])
    for dataset, (label, _) in DATASETS.items():
        members = {row["category"]: row for row in categories if row["dataset"] == dataset
                   and row["cohort"] == "test" and _parameters_match(row, beta, coefficient)}
        if not members:
            lines.extend(["### " + label, "", "本组缺少推荐组合的类别统计。", ""])
            continue
        lines.extend(["### " + label, "", "| 类别 | 数量 | near | line | 未折扣R | 折扣G |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for name in CATEGORIES:
            if name in members:
                row = members[name]
                values = [_number(row[field]) for field in (
                    "near_return_mean", "line_return_mean", "total_return_mean", "discounted_return_mean")]
                lines.append("| %s | %d | %s |" % (CHINESE_CATEGORIES[name], row["episodes"], " | ".join(values)))
        for field, label in (("total_return_mean", "未折扣"), ("discounted_return_mean", "折扣")):
            ordered = sorted(members, key=lambda name: float(members[name][field]), reverse=True)
            lines.extend(["", "%s类别均值排序为 %s。" % (
                label, " > ".join(CHINESE_CATEGORIES[name] for name in ordered))])
        missing = [CHINESE_CATEGORIES[name] for name in CATEGORIES if name not in members]
        if missing:
            lines.extend(["", "本组缺少%s真实样本。" % "、".join(missing)])
        lines.append("")


def _write_report(output, summaries, categories, recommendation, manifest, plots):
    beta, coefficient = recommendation.get("recommended_beta"), recommendation.get("recommended_k")
    lines = ["# 校正near与线段成本配比离线标定", "", "## 结果与推荐", ""]
    if beta is None or coefficient is None:
        lines.append("本轮推荐状态为%s。扫描结果和量级取舍保存在完整表格中。" % recommendation.get("status", "未单列"))
    else:
        lines.append("本轮离线候选为β=%g、k=%g。该组合用于后续受控训练验证。" % (beta, coefficient))
    lines.extend(["", *["- " + str(value) for value in recommendation.get("rationale", [])], ""])
    stronger = recommendation.get("stronger_comparison")
    if stronger:
        lines.extend(["加强对照采用β=%g、k=%g，以相同k检查near放大的影响。" % (
            stronger["beta"], stronger["k"]), ""])
    selected = [dict(beta=beta, k=coefficient)] if beta is not None and coefficient is not None else []
    if stronger:
        selected.append(stronger)
    if selected:
        lines.extend(["### 失败near的经验量级与理论上界", "",
            r"\[\sum_t r^{active}_{near,t}\leq\beta e^{-R_{capture}/\lambda}\quad\text{for a failed episode}\]", "",
            "| 组合 | β | k | 本批失败near最大值 | 任意失败near理论上界 |",
            "| --- | ---: | ---: | ---: | ---: |"])
        for index, parameters in enumerate(selected):
            diagnostic = next((row for row in recommendation.get("diagnostics", [])
                               if _parameters_match(row, parameters["beta"], parameters["k"])), {})
            lines.append("| %s | %g | %g | %s | %s |" % (
                "主候选" if index == 0 else "加强对照", parameters["beta"], parameters["k"],
                _number(diagnostic.get("failure_near_max")),
                _number(diagnostic.get("failure_near_theoretical_bound"))))
        lines.extend(["", "经验最大值描述本批固定轨迹，理论上界描述允许的任意失败回合。提高β会放大理论上界，经验最大值小于约80分仍需结合这一上界评估。", ""])
        lines.extend(["### 候选与加强对照的静止配对", "",
            "| 组合 | 数据组 | 全部近失数 | 静止胜R | 静止胜G | 更近近失数 | 静止胜更近R | 静止胜更近G |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for index, parameters in enumerate(selected):
            for dataset, (label, _) in DATASETS.items():
                row = next((row for row in summaries if row["dataset"] == dataset and row["cohort"] == "test"
                            and _parameters_match(row, parameters["beta"], parameters["k"])), None)
                if row is not None:
                    lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                        "主候选" if index == 0 else "加强对照", label, _number(row.get("near_count"), 0),
                        _percent(row.get("stationary_outrank_near_fraction")),
                        _percent(row.get("stationary_discounted_outrank_near_fraction")),
                        _number(row.get("closer_near_count"), 0),
                        _percent(row.get("stationary_outrank_closer_fraction")),
                        _percent(row.get("stationary_discounted_outrank_closer_fraction"))))
        lines.append("")
    lines.extend([
        "## 方法与样本", "",
        "本轮输入上轮%s条记录，保留原轨迹ID、类别、别名映射和终止结果。测试轨迹去重后按原B组与当前k=0.1实验分别统计，固定0°历史与静止反事实单独汇总。" % manifest.get("source_rows", "已保存"), "",
        r"\[r_t(\beta,k)=r_{original,t}+\frac{\beta}{\beta_0}r_{active\ near,t}(\beta_0)+\frac{k}{k_0}r_{line,t}(k_0)\]", "",
        "源β₀=%g，源k₀=%g。λ固定为%gm，γ=%g。校正near以冻结我方旧位置为反事实，按实际位移归因，惯性滑行计入贡献。" % (
            manifest["source_beta"], manifest["source_k"], manifest["length_scale_m"], manifest["gamma"]), "",
        "near与line对各自系数线性缩放，因此使用保存的分项及折扣分项重算全部组合。图表为原始分组统计。", "",
        "模型加载0次，新增推理0回合，训练0步。固定轨迹的任务结果用于全部组合，离线回报排序提供后续训练候选依据。", "",
        "β扫描为%s，k扫描为%s。网格属于本轮分析选择，生产训练参数的设定另行记录。" % (
            "、".join("%g" % value for value in manifest.get("beta_grid", [])),
            "、".join("%g" % value for value in manifest.get("k_grid", []))), "",
        "失败shaping约80分沿用此前讨论的量级偏好，作为推荐取舍的参考。推荐状态及完整诊断见recommendation.json。", "",
        "## 配比扫描热图", "", "![配比热图](%s)" % plots[0], ""])
    for dataset, (label, _) in DATASETS.items():
        selected = sorted([row for row in summaries if row["dataset"] == dataset and row["cohort"] == "test"],
                          key=lambda row: (float(row["beta"]), float(row["k"])))
        lines.extend(["## " + label, "",
            "静止胜出率使用同角度且真实最小距离优于静止的Near miss子集。G列采用训练折扣系数。", "",
            "| β | k | 近失near | 近失line | 失败near最大值 | 静止胜R | 静止胜G | Near−10–30m的G |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in selected:
            lines.append("| %g | %g | %s | %s | %s | %s | %s | %s |" % (
                row["beta"], row["k"], _number(row.get("near_mean")), _number(row.get("line_mean")),
                _number(row.get("failure_near_max")), _percent(row.get("stationary_outrank_closer_fraction")),
                _percent(row.get("stationary_discounted_outrank_closer_fraction")),
                _number(row.get("near_moderate_discounted_gap"))))
        lines.append("")
    _candidate_tables(lines, categories, recommendation)
    if stronger:
        _candidate_tables(lines, categories, dict(recommended_beta=stronger["beta"], recommended_k=stronger["k"]),
                          title="加强对照的类别回报")
    if len(plots) > 1:
        lines.extend(["## 推荐与基线对比图", "", "![候选与基线回报](%s)" % plots[1], ""])
    lines.extend(["## 解释范围与复现", "",
        "静止near在正系数缩放后仍为0。line依赖基地—敌机闭线段，基地端点具有低偏离成本，因此配比还需结合前出距离和实际训练行为判断。", "",
        "当前样本覆盖的类别、相同角度配对和逐轨迹极值均单独保留。类别平均排序与每条轨迹的排序具有不同统计含义。", "",
        "更近Near miss子集与全部Near miss分别记录静止胜出率，完整统计用于检查配对集合变化后的回报取舍，见combination_summary.csv。", "",
        "500个角度场景已参与奖励开发，这批数据用于冻结轨迹的配比诊断。捕获率和策略保持效果需要后续独立训练及测试验证。", "",
        "[完整组合汇总](combination_summary.csv) · [逐类别统计](category_statistics.csv) · [全部逐回合配比重放](episode_grid_replay.csv)", "",
        "[分析配置](configuration_snapshot.yaml) · [推荐与诊断](recommendation.json) · [数值与来源核验](verification.json)", ""])
    filename = "ACTIVE_NEAR_LINE_CALIBRATION_REPORT.md"
    with (output / filename).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return filename


def write_results(output, summaries, categories, recommendation, manifest, config):
    """Render the actual calibration statistics and the selected candidate."""
    output = Path(output)
    if (output / "ACTIVE_NEAR_LINE_CALIBRATION_REPORT.md").exists():
        raise FileExistsError("Choose a fresh calibration report directory")
    (output / "plots").mkdir(parents=True, exist_ok=True)
    plots = [_heatmaps(output, summaries, manifest)]
    candidate = _candidate_plot(output, categories, recommendation, manifest)
    if candidate is not None:
        plots.append(candidate)
    report = _write_report(output, summaries, categories, recommendation, manifest, plots)
    lines = ["# 校正near与线段成本配比标定", "", "[完整报告](%s)" % report, "",
        "- [配比扫描热图](%s)" % plots[0], "- [完整组合汇总](combination_summary.csv)",
        "- [逐类别统计](category_statistics.csv)", "- [全部逐回合配比重放](episode_grid_replay.csv)",
        "- [推荐与诊断](recommendation.json)", "- [数值与来源核验](verification.json)", ""]
    if candidate is not None:
        lines.append("- [候选与基线回报](%s)" % candidate)
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return dict(report=report, plots=plots)
