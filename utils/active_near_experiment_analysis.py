"""Matched task and flight diagnostics for the active-near training A/B run."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.formal_ppo_evaluation import write_csv, write_json
from utils.formal_trajectory_overview import render_overview
from utils.line_cost_experiment_analysis import _rows, _json, _captured, _mean
from utils.reward_ab_analysis import distance_bin


def _components(rows):
    original = np.asarray([float(r["original_return"]) for r in rows])
    near = np.asarray([float(r["near_shaping_return"]) for r in rows])
    line = np.asarray([float(r.get("line_cost_return", 0)) for r in rows])
    np.testing.assert_allclose(original + near + line,
                               [float(r["episode_return"]) for r in rows], rtol=0, atol=1e-8)
    result = dict(original_four_term_mean=float(original.mean()), near_mean=float(near.mean()),
                  line_mean=float(line.mean()), total_return_mean=_mean(rows, "episode_return"))
    for field in ("near_shaping_uncorrected_return", "near_shaping_removed_return"):
        result[field + "_mean"] = _mean(rows, field) if field in rows[0] else None
    return result


def _collect(label, root, seeds):
    training, evaluation, behavior, validation, learning = [], [], [], [], []
    for seed in seeds:
        run = root / ("seed_%d" % seed)
        episodes, checks = _rows(run / "episodes.csv"), _rows(run / "validation.csv")
        summary = _json(run / "summary.json")
        peak = max(float(r["capture_rate"]) for r in checks)
        positive = [r for r in checks if float(r["capture_rate"]) > 0]
        training.append(dict(group=label, seed=seed, steps=summary["env_steps"],
            episodes=len(episodes), captures=sum(_captured(r) for r in episodes),
            capture_rate=sum(_captured(r) for r in episodes) / len(episodes),
            min_distance_mean=_mean(episodes, "minimum_distance"),
            first_positive_validation=int(positive[0]["env_steps"]) if positive else None,
            best_validation_capture_rate=peak, best_step=summary["best_env_steps"],
            final_validation_capture_rate=float(checks[-1]["capture_rate"]),
            peak_to_final_drop=peak - float(checks[-1]["capture_rate"]), **_components(episodes)))
        validation.extend(dict(group=label, seed=seed, **r) for r in checks)
        learning.extend(dict(group=label, seed=seed, **r) for r in _rows(run / "updates.csv"))
        environment = yaml.safe_load((run / "environment_config.yaml").read_text(encoding="utf-8"))
        base = np.asarray(environment["mission"]["base_position"])
        attack_radius = float(environment["mission"]["attack_radius"])
        for model, folder in (("best", "final_test"), ("final", "final_model_test")):
            directory = run / folder
            rows = _rows(directory / "episodes.csv")
            metrics = _json(directory / "summary.json")
            bins = {name: 0 for name in ("lt5", "5_to_10", "10_to_30", "30_to_100", "gt100")}
            paths, farthest, near_base, transverse, vertical, initial_y = [], [], [], [], [], []
            for row in rows:
                bins[distance_bin(float(row["min_distance"]))] += 1
                with np.load(directory / row["trajectory"], allow_pickle=False) as data:
                    if not all(np.isfinite(data[name]).all() for name in ("states", "rewards", "actions")):
                        raise FloatingPointError("Non-finite saved evaluation")
                    states = data["states"]
                    own = states[:, 0, :3]
                    radius = np.linalg.norm(own - base, axis=1)
                    paths.append(float(np.linalg.norm(np.diff(own, axis=0), axis=1).sum()))
                    farthest.append(float(radius.max()))
                    near_base.append(float(np.mean(radius < attack_radius)))
                    relative = states[:, 0, :3] - states[:, 1, :3]
                    closest = int(np.argmin(np.linalg.norm(relative, axis=1)))
                    angle = np.deg2rad(float(row["angle_degrees"]))
                    transverse.append(abs(float(-np.sin(angle) * relative[closest, 0]
                                                + np.cos(angle) * relative[closest, 1])))
                    vertical.append(abs(float(relative[closest, 2])))
                    initial_y.append(float(data["actions"][0, 1]))
            negative = [r for r in rows if float(r["angle_degrees"]) < 0]
            positive_angles = [r for r in rows if float(r["angle_degrees"]) > 0]
            evaluation.append(dict(group=label, seed=seed, model=model, episodes=len(rows),
                captures=metrics["captures"], capture_rate=metrics["capture_rate"],
                defense_failure_rate=metrics["defense_failure_rate"], timeout_rate=metrics["timeout_rate"],
                average_capture_time=metrics["average_interception_time"], min_distance_mean=metrics["average_min_distance"],
                negative_capture_rate=float(np.mean([_captured(r) for r in negative])),
                positive_capture_rate=float(np.mean([_captured(r) for r in positive_angles])),
                **_components(rows), **bins))
            angles, actions = np.asarray([float(r["angle_degrees"]) for r in rows]), np.asarray(initial_y)
            behavior.append(dict(group=label, seed=seed, model=model,
                traveled_distance_mean=float(np.mean(paths)), maximum_base_distance_mean=float(np.mean(farthest)),
                near_base_state_fraction_mean=float(np.mean(near_base)),
                closest_transverse_error_mean=float(np.mean(transverse)), closest_vertical_error_mean=float(np.mean(vertical)),
                initial_y_action_mean_negative=float(actions[angles < 0].mean()),
                initial_y_action_mean_positive=float(actions[angles > 0].mean())))
    return training, evaluation, behavior, validation, learning


def _write_union(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, [{key: row.get(key, "") for key in keys} for row in rows])


def analyze(output_root, group_outputs, baseline_root, posts):
    output = Path(output_root)
    directory = output / "comparison"
    directory.mkdir(exist_ok=True)
    first = Path(next(iter(group_outputs.values())))
    settings = yaml.safe_load((first / "experiment_config.yaml").read_text(encoding="utf-8"))
    seeds = settings["experiment"]["seeds"]
    collections = [[], [], [], [], []]
    for label, root in [("previous_k0p1", Path(baseline_root))] + list(group_outputs.items()):
        for target, values in zip(collections, _collect(label, Path(root), seeds)):
            target.extend(values)
    training, evaluation, behavior, validation, learning = collections
    for filename, rows in (("training_comparison.csv", training), ("evaluation_comparison.csv", evaluation),
                           ("behavior_comparison.csv", behavior), ("validation_history.csv", validation),
                           ("learning_metrics.csv", learning)):
        _write_union(directory / filename, rows)
    figure, axes = plt.subplots(1, len(seeds), figsize=(18, 4.5), squeeze=False)
    for column, seed in enumerate(seeds):
        axis = axes[0, column]
        for label in ["previous_k0p1"] + list(group_outputs):
            rows = [r for r in validation if r["group"] == label and r["seed"] == seed]
            axis.plot([int(r["env_steps"]) for r in rows], [100 * float(r["capture_rate"]) for r in rows], marker=".", label=label)
        axis.set(title="Seed %d" % seed, xlabel="Aggregate training steps", ylabel="Validation capture (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "validation_capture_comparison.png", dpi=170)
    plt.close(figure)
    for label, root in group_outputs.items():
        render_overview(posts[label], Path(root) / "trajectory_evolution")
    lines = ["# 主动接近near与低线段成本的PPO训练对照", "",
        "两组分别采用β100/k0.01、β200/k0.01，λ20m。每组1001、1002、1003各从零300000步，8环境，±30°均匀来袭。", "",
        "本次沿用最近三个Seed协议，共六次独立初始化训练，总预算1800000 aggregate environment steps，顺序执行。", "",
        "near采用冻结我方本步旧位置的反事实，只对突破实际历史最小距离且优于反事实的我方位移发奖，惯性位移计入。", "",
        "每10k验证相同100角度；Best按捕获率、失守率、捕获时间、最小距离选择；Best与Final各使用相同500角度测试。", "",
        "## 训练与验证", "",
        "| 组别 | Seed | 训练捕获 | 训练捕获率 | 最佳验证 | Best步数 | 最终验证 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in training:
        lines.append("| {group} | {seed} | {captures}/{episodes} | {capture_rate:.2%} | {best_validation_capture_rate:.1%} | {best_step} | {final_validation_capture_rate:.1%} |".format(**r))
    lines.extend(["", "![验证趋势](validation_capture_comparison.png)", "", "## 500场景确定性测试", "",
        "| 组别 | Seed | 模型 | 捕获率 | 负角捕获率 | 正角捕获率 | 最小距离/m | 原四项回报 | near | line |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in evaluation:
        lines.append("| {group} | {seed} | {model} | {capture_rate:.1%} | {negative_capture_rate:.1%} | {positive_capture_rate:.1%} | {min_distance_mean:.3f} | {original_four_term_mean:.3f} | {near_mean:.3f} | {line_mean:.3f} |".format(**r))
    lines.extend(["", "## 路径与交会误差", "",
        "| 组别 | Seed | 模型 | 路径/m | 最大前出/m | 基地10m内状态比例 | 交会横向误差/m | 交会高度误差/m |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for r in behavior:
        lines.append("| {group} | {seed} | {model} | {traveled_distance_mean:.2f} | {maximum_base_distance_mean:.2f} | {near_base_state_fraction_mean:.1%} | {closest_transverse_error_mean:.3f} | {closest_vertical_error_mean:.3f} |".format(**r))
    lines.extend(["", "## 奖励与训练解释", "",
        "总回报定义随β及near归因变化，共同原四项和任务结果分别保存，组间结论结合捕获率、保持性与实际路径判断。", "",
        "β200任意失败near的理论上界约155.76分，β100约77.88分，实际分项极值另从原始日志核查。", "",
        "0°视频与演化展示共同单场景，整体性能使用500场景统计。相同500角度已参与离线奖励标定，此处用于受控诊断。", "",
        "每组trajectory_evolution目录覆盖全部已保存唯一Actor策略，按实际更新步数由浅到深，三Seed并排；源checkpoint别名另存。", "",
        "[训练分项](training_comparison.csv) · [任务指标与距离分布](evaluation_comparison.csv) · [路径与误差](behavior_comparison.csv)", "",
        "[完整验证历史](validation_history.csv) · [PPO指标](learning_metrics.csv)", ""])
    report = directory / "ACTIVE_NEAR_TRAINING_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    navigation = ["# 主动接近near配比训练", "", "[综合对照报告](comparison/ACTIVE_NEAR_TRAINING_REPORT.md)", "",
        "两组各三个Seed从零300k，共180万训练步。β100与β200分别保存，k均为0.01。", "",
        "| 组别 | 三Seed密集演化 | 分阶段演化 |", "| --- | --- | --- |"]
    for label, root in group_outputs.items():
        relative = Path(root).relative_to(output).as_posix()
        navigation.append("| %s | [XY](%s/trajectory_evolution/trajectory_evolution_overview_xy.png) | [阶段图](%s/trajectory_evolution/trajectory_evolution_phases_xy.png) |" % (label, relative, relative))
    navigation.extend(["", "| 组别 | Seed | 训练报告 | Best视频 | Final视频 |", "| --- | ---: | --- | --- | --- |"])
    for label, entries in posts.items():
        for post in entries:
            relative = Path(post["output_dir"]).relative_to(output).as_posix()
            seed = Path(post["output_dir"]).parent.name
            navigation.append("| %s | %s | [报告](%s/TRAINING_REPORT.md) | [Best](%s/videos/best/best_full_episode.mp4) | [Final](%s/videos/final/final_full_episode.mp4) |" % (label, seed, relative, relative, relative))
    (output / "README.md").write_text("\n".join(navigation) + "\n", encoding="utf-8")
    write_json(directory / "summary.json", dict(training=training, evaluation=evaluation, behavior=behavior))
    return report.resolve()
