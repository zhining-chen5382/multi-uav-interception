"""Matched diagnostics for the independent line-cost training experiment."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.formal_ppo_evaluation import write_csv, write_json
from utils.reward_ab_analysis import distance_bin


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _captured(row):
    return str(row["captured"]).lower() == "true"


def _mean(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def trajectory_overview(output, posts):
    from utils.formal_trajectory_overview import render_overview
    return render_overview(posts, output)


def analyze(output, baseline_root, posts):
    directory = output / "comparison"
    directory.mkdir(exist_ok=True)
    seeds = [int(Path(post["output_dir"]).parent.name.split("_")[1]) for post in posts]
    settings = yaml.safe_load((output / ("seed_%d" % seeds[0]) / "training_config.yaml").read_text(encoding="utf-8"))
    training_rows, evaluation_rows, behavior_rows, validation_rows = [], [], [], []
    for label, root in (("previous_B", baseline_root), ("k0p1", output)):
        for seed in seeds:
            run = root / ("seed_%d" % seed)
            environment = yaml.safe_load((run / "environment_config.yaml").read_text(encoding="utf-8"))
            base = np.asarray(environment["mission"]["base_position"])
            attack_radius = environment["mission"]["attack_radius"]
            episodes = _rows(run / "episodes.csv")
            validation = _rows(run / "validation.csv")
            summary = _json(run / "summary.json")
            original = np.asarray([float(row["original_return"]) for row in episodes])
            near = np.asarray([float(row["near_shaping_return"]) for row in episodes])
            line = np.asarray([float(row.get("line_cost_return", 0)) for row in episodes])
            np.testing.assert_allclose(original + near + line,
                                       [float(row["episode_return"]) for row in episodes], rtol=0, atol=1e-8)
            positive = [row for row in validation if float(row["capture_rate"]) > 0]
            peak = max(float(row["capture_rate"]) for row in validation)
            training_rows.append(dict(group=label, seed=seed, steps=summary["env_steps"],
                episodes=len(episodes), captures=sum(_captured(row) for row in episodes),
                capture_rate=sum(_captured(row) for row in episodes) / len(episodes),
                original_return_mean=float(original.mean()), near_return_mean=float(near.mean()),
                line_cost_mean=float(line.mean()), common_B_return_mean=float((original + near).mean()),
                logged_return_mean=_mean(episodes, "episode_return"),
                min_distance_mean=_mean(episodes, "minimum_distance"),
                first_positive_validation=int(positive[0]["env_steps"]) if positive else None,
                best_validation_capture_rate=peak, best_step=summary["best_env_steps"],
                final_validation_capture_rate=float(validation[-1]["capture_rate"]),
                peak_to_final_drop=peak - float(validation[-1]["capture_rate"])))
            validation_rows.extend(dict(group=label, seed=seed, **row) for row in validation)
            for role, folder in (("best", "final_test"), ("final", "final_model_test")):
                evaluation = run / folder
                rows = _rows(evaluation / "episodes.csv")
                metrics = _json(evaluation / "summary.json")
                dist_counts = {name: 0 for name in ("lt5", "5_to_10", "10_to_30", "30_to_100", "gt100")}
                radius_fractions, traveled, maximum_radius = [], [], []
                initial_actions, angles = [], []
                for row in rows:
                    dist_counts[distance_bin(float(row["min_distance"]))] += 1
                    with np.load(evaluation / row["trajectory"], allow_pickle=False) as data:
                        own = data["states"][:, 0, :3]
                        radius = np.linalg.norm(own - base, axis=1)
                        radius_fractions.append(float(np.mean(radius < attack_radius)))
                        maximum_radius.append(float(radius.max()))
                        traveled.append(float(np.linalg.norm(np.diff(own, axis=0), axis=1).sum()))
                        initial_actions.append(float(data["actions"][0, 1]))
                    angles.append(float(row["angle_degrees"]))
                item = dict(group=label, seed=seed, model=role, episodes=len(rows),
                    captures=metrics["captures"], capture_rate=metrics["capture_rate"],
                    defense_failure_rate=metrics["defense_failure_rate"], timeout_rate=metrics["timeout_rate"],
                    average_capture_time=metrics["average_interception_time"],
                    min_distance_mean=metrics["average_min_distance"],
                    original_return_mean=_mean(rows, "original_return"),
                    near_return_mean=_mean(rows, "near_shaping_return"),
                    line_cost_mean=float(np.mean([float(row.get("line_cost_return", 0)) for row in rows])),
                    negative_capture_rate=float(np.mean([_captured(row) for row in rows if float(row["angle_degrees"]) < 0])),
                    positive_capture_rate=float(np.mean([_captured(row) for row in rows if float(row["angle_degrees"]) > 0])),
                    **dist_counts)
                evaluation_rows.append(item)
                actions, angles = np.asarray(initial_actions), np.asarray(angles)
                behavior_rows.append(dict(group=label, seed=seed, model=role,
                    near_base_state_fraction_mean=float(np.mean(radius_fractions)),
                    traveled_distance_mean=float(np.mean(traveled)),
                    maximum_base_distance_mean=float(np.mean(maximum_radius)),
                    initial_y_action_mean_negative=float(actions[angles < 0].mean()),
                    initial_y_action_mean_positive=float(actions[angles > 0].mean())))
    write_csv(directory / "training_comparison.csv", training_rows)
    write_csv(directory / "evaluation_comparison.csv", evaluation_rows)
    write_csv(directory / "behavior_comparison.csv", behavior_rows)
    # The line-enabled validation log may have additional diagnostic columns.
    keys = list(dict.fromkeys(key for row in validation_rows for key in row))
    write_csv(directory / "validation_history.csv", [{key: row.get(key, "") for key in keys} for row in validation_rows])
    figure, axes = plt.subplots(1, len(seeds), figsize=(15, 4), squeeze=False)
    for column, seed in enumerate(seeds):
        for group in ("previous_B", "k0p1"):
            rows = [row for row in validation_rows if row["seed"] == seed and row["group"] == group]
            axes[0, column].plot([int(row["env_steps"]) for row in rows],
                                 [100 * float(row["capture_rate"]) for row in rows], marker=".", label=group)
        axes[0, column].set(title="Seed %d" % seed, xlabel="Aggregate training steps", ylabel="Validation capture (%)")
        axes[0, column].grid(alpha=.2)
        axes[0, column].legend()
    figure.tight_layout()
    figure.savefig(directory / "validation_capture_comparison.png", dpi=160)
    plt.close(figure)
    trajectory_overview(directory, posts)
    lines = ["# k=0.1线段成本训练对照", "",
        "%d个Seed各从零完成%d步，%d环境，每%d步验证%d角度。" % (
            len(seeds), settings["training"]["total_env_steps"], settings["training"]["num_envs"],
            settings["training"]["evaluation_interval_steps"], settings["training"]["validation_episodes"]),
        "新增项为−0.1乘三维闭线段距离再乘dt，在原四项和β100、λ20m历史最小距离奖励上累加。",
        "实验规模沿用上一轮。固定物理归一化16维、网络、PPO与动力学见配置快照。", "",
        "## 训练与验证", "",
        "| 组别 | Seed | 训练捕获 | 训练捕获率 | 最高验证 | Best步数 | 最终验证 | 平均线段奖励 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    if settings["experiment"].get("engineering_fixture"):
        lines[2:2] = ["**工程fixture，数值用于检验保存、评估与制图流程。**", ""]
    for row in training_rows:
        lines.append("| {group} | {seed} | {captures}/{episodes} | {rate:.2%} | {best:.1%} | {best_step} | {final:.1%} | {line:.3f} |".format(
            **row, rate=row["capture_rate"], best=row["best_validation_capture_rate"],
            final=row["final_validation_capture_rate"], line=row["line_cost_mean"]))
    lines.extend(["", "![验证捕获率](validation_capture_comparison.png)", "",
        "## 相同500场景的确定性比较", "",
        "| 组别 | Seed | 模型 | 捕获率 | 负角捕获率 | 正角捕获率 | 平均最小距离m | 平均线段奖励 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in evaluation_rows:
        lines.append("| {group} | {seed} | {model} | {rate:.1%} | {negative:.1%} | {positive:.1%} | {distance:.3f} | {line:.3f} |".format(
            **row, rate=row["capture_rate"], negative=row["negative_capture_rate"],
            positive=row["positive_capture_rate"], distance=row["min_distance_mean"], line=row["line_cost_mean"]))
    lines.extend(["", "## 路径与基地附近停留", "",
        "停留指标为整回合记录状态中，我方距基地小于10m的比例。路径长度与最大前出半径单独保存，帮助区分守基地与主动运动。", "",
        "| 组别 | Seed | 模型 | 基地10m内状态比例 | 平均路径m | 平均最大前出m |",
        "| --- | ---: | --- | ---: | ---: | ---: |"])
    for row in behavior_rows:
        lines.append("| {group} | {seed} | {model} | {near:.1%} | {path:.2f} | {radius:.2f} |".format(
            **row, near=row["near_base_state_fraction_mean"], path=row["traveled_distance_mean"],
            radius=row["maximum_base_distance_mean"]))
    lines.extend(["", "![真实采样步数的XY演化](trajectory_evolution_overview_xy.png)", "",
        "总览覆盖各Seed全部已保存的唯一Actor策略，以实际更新步数着色。阶段分图见trajectory_evolution_phases_xy.png，检查点别名映射见checkpoint_aliases.csv。完整指标见trajectory_evolution_sampled_metrics.csv。", "",
        "## 解读范围", "",
        "Capture沿用三维距离严格小于5m。基地攻击距离严格小于10m，捕获优先。视频使用同一固定0°场景。",
        "原B与本轮总奖励定义不同，共同B回报为原四项加near，line分量单独保存。",
        "配对500场景已参与此前奖励离线开发，本次结果用于受控诊断比较，后续独立泛化检验宜另设新场景。", "",
        "[训练比较](training_comparison.csv) · [完整测试指标与距离分布](evaluation_comparison.csv) · [运动与方向响应](behavior_comparison.csv)", ""])
    report = directory / "LINE_COST_K0P1_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    navigation = ["# k=0.1线段成本训练结果", "", "[综合对照报告](comparison/LINE_COST_K0P1_REPORT.md)", "",
        "[XY演化总览](comparison/trajectory_evolution_overview_xy.png)", "",
        "| Seed | 训练报告 | 演化图 | Best视频 | Final视频 |", "| --- | --- | --- | --- | --- |"]
    for seed, post in zip(seeds, posts):
        relative = Path(post["output_dir"]).relative_to(output).as_posix()
        navigation.append("| %d | [报告](%s/TRAINING_REPORT.md) | [XY](%s/trajectory_evolution/trajectory_evolution_xy.png) | [Best](%s/videos/best/best_full_episode.mp4) | [Final](%s/videos/final/final_full_episode.mp4) |" % (
            seed, relative, relative, relative, relative))
    (output / "README.md").write_text("\n".join(navigation) + "\n", encoding="utf-8")
    return report
