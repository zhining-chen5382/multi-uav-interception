"""Recalculate proposed distance shaping using saved formal-PPO trajectories.

This module imports neither the environment nor a learning algorithm. Distances
include the initial state. Original rewards use the post-step distance, while
the proposed term is the unmodified difference of two consecutive potentials.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from utils.artifact_paths import resolve_artifact_path


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("capture", "near_miss", "moderate_failure", "other_failure", "clear_failure")
ORDERED_CATEGORIES = ("capture", "near_miss", "moderate_failure", "clear_failure")
COHORTS = ("held_out_best_test", "fixed_zero_checkpoint_history")


def _distances(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Distances must be a finite nonnegative vector including the initial state")
    return values


def reward_components(distances, dt, captured, failed, coefficients):
    distances = _distances(distances)
    if not np.isfinite(dt) or dt <= 0 or (captured and failed):
        raise ValueError("Invalid time step or mutually exclusive terminal outcomes")
    values = np.asarray(list(coefficients.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Reward coefficients must be finite")
    count = len(distances) - 1
    components = dict(
        distance_cost=-coefficients["distance_cost_coefficient"] * distances[1:] * dt,
        survival=np.full(count, coefficients["survival_penalty_per_second"] * dt),
        capture=np.zeros(count), defense_failure=np.zeros(count))
    components["capture"][-1] = coefficients["capture_reward"] if captured else 0.0
    components["defense_failure"][-1] = coefficients["attack_failure_penalty"] if failed else 0.0
    components["original"] = sum(components.values())
    return components


def near_shaping(distances, length_scale_m, beta):
    distances = _distances(distances)
    if not np.isfinite(length_scale_m) or length_scale_m <= 0 or not np.isfinite(beta) or beta < 0:
        raise ValueError("Length scale must be positive and beta nonnegative")
    return beta * np.diff(np.exp(-distances / length_scale_m))


def discounted_return(rewards, gamma):
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not len(rewards) or not np.isfinite(rewards).all() or not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Invalid reward vector or discount factor")
    return float(np.dot(gamma ** np.arange(len(rewards)), rewards))


def category(captured, min_distance, near_miss_max=10.0, moderate_max=30.0, clear_min=100.0):
    if captured:
        return "capture"
    if min_distance <= near_miss_max:
        return "near_miss"
    if min_distance <= moderate_max:
        return "moderate_failure"
    if min_distance > clear_min:
        return "clear_failure"
    return "other_failure"


def read_csv(path):
    with resolve_artifact_path(path, root=ROOT).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(resolve_artifact_path(path, root=ROOT).read_bytes()).hexdigest()


def _track(path, hashes):
    path = resolve_artifact_path(path, root=ROOT)
    hashes[str(path)] = sha256(path)
    return path


def load_sources(config, hashes):
    root = resolve_artifact_path(config["source_root"], root=ROOT)
    experiment_path = _track(root / "experiment_config.yaml", hashes)
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    gamma = float(experiment["ppo"]["gamma"])
    episodes = []
    for seed in config["seeds"]:
        run = root / ("seed_%d" % seed)
        summary_path = _track(run / "summary.json", hashes)
        best_step = json.loads(summary_path.read_text(encoding="utf-8"))["best_env_steps"]
        test = run / "final_test"
        environment_path = _track(test / "environment_config.yaml", hashes)
        environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
        index = _track(test / "episodes.csv", hashes)
        for row in read_csv(index):
            episodes.append(_load_episode(test / row["trajectory"], environment, row,
                "held_out_best_test", seed, best_step, config, hashes))
        post = run / config["postprocessing"][seed]
        index = _track(post / "trajectory_evolution/checkpoint_metrics.csv", hashes)
        seen = set()
        for row in read_csv(index):
            if row["actor_sha256"] in seen:
                continue
            seen.add(row["actor_sha256"])
            episodes.append(_load_episode(post / row["trajectory"], environment, row,
                "fixed_zero_checkpoint_history", seed, int(row["step"]), config, hashes))
    return episodes, gamma


def _load_episode(path, environment, index_row, cohort, seed, step, config, hashes):
    path = _track(path, hashes)
    with np.load(path, allow_pickle=False) as archive:
        states, time, saved_rewards = archive["states"], archive["time"], archive["rewards"]
        metadata = json.loads(archive["metadata_json"].item())
    dt = float(environment["environment"]["dt"])
    if states.shape != (len(saved_rewards) + 1, 2, 6) or not np.isfinite(states).all():
        raise ValueError("Invalid complete trajectory " + str(path))
    np.testing.assert_allclose(time, np.arange(len(states)) * dt, rtol=0, atol=1e-10)
    distances = np.linalg.norm(states[:, 0, :3] - states[:, 1, :3], axis=1)
    base_distances = np.linalg.norm(states[:, 1, :3] - environment["mission"]["base_position"], axis=1)
    captured, failed = metadata["captured"], metadata["defense_failure"]
    components = reward_components(distances, dt, captured, failed, environment["reward"])
    np.testing.assert_allclose(components["original"], saved_rewards, rtol=0, atol=1e-10)
    np.testing.assert_allclose(components["original"].sum(), metadata["episode_return"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(distances.min(), float(index_row["min_distance"]), rtol=0, atol=1e-8)
    if np.any(distances[1:-1] < environment["capture"]["radius"]):
        raise ValueError("Trajectory continued after a capture")
    if bool(distances[-1] < environment["capture"]["radius"]) != bool(captured):
        raise ValueError("Stored capture outcome disagrees with distance")
    if bool(not captured and base_distances[-1] < environment["mission"]["attack_radius"]) != bool(failed):
        raise ValueError("Stored failure outcome disagrees with distance")
    return dict(
        episode_id="%s_seed%d_step%d_scene%d" % (cohort, seed, step, metadata["scenario_id"]),
        cohort=cohort, training_seed=seed, checkpoint_step=step,
        environment_seed=metadata["seed"], angle_degrees=metadata["angle_degrees"],
        category=category(captured, float(distances.min()), config["near_miss_max_m"],
                          config["moderate_failure_max_m"], config["clear_failure_min_m"]),
        captured=captured, defense_failure=failed, timeout=metadata["timeout"],
        episode_length=len(saved_rewards), event_time=time[-1],
        min_distance=float(distances.min()), final_distance=float(distances[-1]),
        initial_distance=float(distances[0]), final_attacker_base_distance=float(base_distances[-1]),
        simultaneous_capture_and_attack_zone=bool(captured and base_distances[-1] < environment["mission"]["attack_radius"]),
        trajectory=str(path), trajectory_sha256=hashes[str(path)],
        dt=dt, distances=distances, time=time, components=components,
        max_original_reward_error=float(np.max(np.abs(components["original"] - saved_rewards))))


def replay(episodes, length_scales, betas, gamma, target_range):
    rows = []
    for episode in episodes:
        original = episode["components"]["original"]
        for length_scale in length_scales:
            for beta in betas:
                near = near_shaping(episode["distances"], length_scale, beta)
                total = float(near.sum())
                potential = np.exp(-episode["distances"] / length_scale)
                np.testing.assert_allclose(total, beta * (potential[-1] - potential[0]), rtol=0, atol=1e-11)
                discounted = discounted_return(near, gamma)
                telescoped = beta * (-potential[0] + gamma ** (len(near) - 1) * potential[-1]
                    + (1 - gamma) * np.dot(gamma ** np.arange(len(near) - 1), potential[1:-1]))
                np.testing.assert_allclose(discounted, telescoped, rtol=0, atol=1e-11)
                delta = np.diff(episode["distances"])
                if np.any(near[delta > 0] > 0) or np.any(near[delta == 0] != 0):
                    raise ValueError("Invalid shaping sign")
                rows.append(dict(
                    episode_id=episode["episode_id"], cohort=episode["cohort"],
                    training_seed=episode["training_seed"], checkpoint_step=episode["checkpoint_step"],
                    angle_degrees=episode["angle_degrees"], category=episode["category"],
                    lambda_m=length_scale, beta=beta,
                    original_return=float(original.sum()), new_return=float((original + near).sum()),
                    near_return=total, near_positive_sum=float(near[near > 0].sum()),
                    near_negative_sum=float(near[near < 0].sum()),
                    peak_cumulative_near=float(np.max(np.r_[0, np.cumsum(near)])),
                    original_discounted_return=discounted_return(original, gamma),
                    new_discounted_return=discounted_return(original + near, gamma),
                    discounted_near_return=discounted,
                    near_in_target_range=bool(target_range[0] <= total <= target_range[1])))
    return rows


def _pairwise_win_rate(higher, lower):
    higher, lower = np.asarray(higher), np.sort(lower)
    if not len(higher) or not len(lower):
        return None
    return float(np.searchsorted(lower, higher, side="left").sum() / (len(higher) * len(lower)))


def summarize(rows, length_scales, betas):
    comparisons, groups, rankings = [], [], []
    for cohort in COHORTS:
        for length_scale in length_scales:
            for beta in betas:
                selected = [row for row in rows if row["cohort"] == cohort
                            and row["lambda_m"] == length_scale and row["beta"] == beta]
                by_category = {name: [row for row in selected if row["category"] == name] for name in CATEGORIES}
                comparison = dict(cohort=cohort, lambda_m=length_scale, beta=beta, episodes=len(selected))
                for name, members in by_category.items():
                    if not members:
                        continue
                    group = dict(cohort=cohort, lambda_m=length_scale, beta=beta, category=name, episodes=len(members))
                    for field in ("original_return", "new_return", "near_return", "original_discounted_return",
                                  "new_discounted_return", "discounted_near_return", "near_positive_sum", "near_negative_sum"):
                        values = np.asarray([row[field] for row in members])
                        for suffix, value in (("mean", values.mean()), ("median", np.median(values)),
                                              ("min", values.min()), ("max", values.max())):
                            group[field + "_" + suffix] = float(value)
                    group["near_target_fraction"] = float(np.mean([r["near_in_target_range"] for r in members]))
                    groups.append(group)
                    comparison[name + "_new_mean"] = group["new_return_mean"]
                    comparison[name + "_discounted_new_mean"] = group["new_discounted_return_mean"]
                    comparison[name + "_near_mean"] = group["near_return_mean"]
                    if name == "capture":
                        comparison["capture_near_target_fraction"] = group["near_target_fraction"]
                captures = by_category["capture"]
                failures = [row for row in selected if row["category"] != "capture"]
                for field in ("original_return", "new_return", "original_discounted_return", "new_discounted_return"):
                    means = {name: float(np.mean([row[field] for row in members])) for name,members in by_category.items() if members}
                    ordered = [name for name in ORDERED_CATEGORIES if name in means]
                    comparison[field + "_mean_order"] = " > ".join(sorted(ordered, key=lambda name: means[name], reverse=True))
                    comparison[field + "_four_group_mean_order_ok"] = bool(len(ordered) == 4 and all(means[a] > means[b] for a,b in zip(ordered, ordered[1:])))
                    comparison[field + "_all_capture_above_failure"] = bool(captures and failures and min(r[field] for r in captures) > max(r[field] for r in failures))
                    for higher, lower in zip(ORDERED_CATEGORIES, ORDERED_CATEGORIES[1:]):
                        rankings.append(dict(cohort=cohort, lambda_m=length_scale, beta=beta, return_type=field,
                            higher_category=higher, lower_category=lower,
                            pairwise_win_rate=_pairwise_win_rate([r[field] for r in by_category[higher]], [r[field] for r in by_category[lower]])))
                comparison["max_failure_new_return"] = max(r["new_return"] for r in failures)
                comparison["max_failure_near_return"] = max(r["near_return"] for r in failures)
                comparison["max_failure_discounted_new_return"] = max(r["new_discounted_return"] for r in failures)
                comparisons.append(comparison)
    # Auxiliary cohort may lack a category, so use a common column schema.
    fields = list(dict.fromkeys(key for row in comparisons for key in row))
    comparisons = [{key: row.get(key) for key in fields} for row in comparisons]
    return comparisons, groups, rankings


def select_representatives(episodes, per_category):
    selected = []
    for name in ORDERED_CATEGORIES:
        candidates = [episode for episode in episodes if episode["cohort"] == COHORTS[0] and episode["category"] == name]
        if not candidates:
            candidates = [episode for episode in episodes if episode["category"] == name]
        median = float(np.median([episode["min_distance"] for episode in candidates]))
        used_seeds, used_angles = set(), []
        for _ in range(min(per_category, len(candidates))):
            candidates.sort(key=lambda episode: (
                episode["training_seed"] in used_seeds,
                any(abs(episode["angle_degrees"] - angle) < 5 for angle in used_angles),
                abs(episode["min_distance"] - median), episode["episode_id"]))
            choice = candidates.pop(0)
            selected.append(choice)
            used_seeds.add(choice["training_seed"])
            used_angles.append(choice["angle_degrees"])
    return selected


def plot_results(episodes, representatives, rows, comparisons, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir()
    palettes = {10.0: "#2976b7", 20.0: "#e1812c", 30.0: "#40935c"}
    styles = {20.0: ":", 40.0: "--", 60.0: "-"}
    records = []
    for number, episode in enumerate(representatives, 1):
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        time, distances = episode["time"], episode["distances"]
        axes[0].plot(time, distances, color="black", label="3D pursuer-attacker distance")
        axes[0].axhline(5, color="green", ls=":", label="Capture radius 5 m")
        axes[0].scatter(time[distances.argmin()], distances.min(), color="red", s=20)
        axes[0].set_ylabel("Distance (m)")
        for name, values in episode["components"].items():
            if name != "original":
                axes[1].plot(time[1:], values, label=name)
        axes[1].set_yscale("symlog", linthresh=0.1)
        axes[1].set_ylabel("Original terms\nsymlog")
        near_values = {}
        for length_scale in config["length_scales_m"]:
            for beta in config["betas"]:
                near = near_shaping(distances, length_scale, beta)
                label = "lambda=%g m, beta=%g" % (length_scale, beta)
                for axis, values in ((axes[2], near), (axes[3], np.cumsum(near))):
                    axis.plot(time[1:], values, color=palettes[length_scale], ls=styles[beta], label=label, lw=1.3)
                near_values["near_lambda%g_beta%g" % (length_scale, beta)] = near
        axes[2].set_ylabel("Near reward / step")
        axes[2].axhline(0, color="gray", lw=0.7)
        axes[3].set_ylabel("Cumulative near reward")
        axes[3].axhline(0, color="gray", lw=0.7)
        axes[3].axhspan(*config["target_capture_shaping_range"], color="green", alpha=0.08, label="30-50 target")
        axes[3].set_xlabel("Episode time (s)")
        for axis in axes:
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8, ncol=4)
        axes[2].legend(fontsize=8, ncol=3)
        title = "%02d | %s | seed %d | angle %.2f deg | min %.3f m" % (
            number, episode["category"], episode["training_seed"], episode["angle_degrees"], episode["min_distance"])
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        filename = "representative_%02d_%s.png" % (number, episode["category"])
        fig.savefig(plots / filename, dpi=150)
        plt.close(fig)
        for index in range(len(time)-1):
            row = dict(episode_id=episode["episode_id"], step=index, time_start=time[index], time_end=time[index+1],
                       distance_before=distances[index], distance_after=distances[index+1])
            row.update({name: values[index] for name, values in episode["components"].items()})
            row.update({name: values[index] for name, values in near_values.items()})
            records.append(row)
        episode["component_plot"] = "plots/" + filename
    write_csv(output / "representative_step_components.csv", records)
    primary = [row for row in comparisons if row["cohort"] == COHORTS[0]]
    labels = ["L%g B%g" % (r["lambda_m"],r["beta"]) for r in primary]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, field in ((axes[0], "_new_mean"), (axes[1], "_discounted_new_mean")):
        for name in CATEGORIES:
            axis.plot(range(len(primary)), [r.get(name + field, np.nan) for r in primary], marker="o", label=name)
        axis.set_xticks(range(len(primary)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_ylabel("Mean return")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_title("Held-out 2,500 real trajectories | undiscounted")
    axes[1].set_title("Held-out 2,500 real trajectories | gamma=0.99")
    fig.tight_layout()
    fig.savefig(plots / "nine_parameter_return_comparison.png", dpi=160)
    plt.close(fig)


def write_report(output, manifest):
    """Write the analysis narrative from the exact saved CSV statistics."""
    comparisons = read_csv(output / "parameter_comparison.csv")
    groups = read_csv(output / "category_return_statistics.csv")
    episodes = read_csv(output / "episode_reward_replay.csv")
    sources = read_csv(output / "trajectory_manifest.csv")
    representatives = read_csv(output / "representatives.csv")
    primary = [row for row in comparisons if row["cohort"] == COHORTS[0]]
    chosen_groups = {row["category"]: row for row in groups if row["cohort"] == COHORTS[0]
                     and float(row["lambda_m"]) == 20 and float(row["beta"]) == 40}
    chosen = {row["episode_id"]: row for row in episodes
              if float(row["lambda_m"]) == 20 and float(row["beta"]) == 40}
    labels = dict(capture="Capture", near_miss="Near miss", moderate_failure="中等失败",
                  other_failure="30–100m其他失败", clear_failure="明显失败")
    lines = ["# Offline near-distance reward replay", "",
        "## 目的与计算定义", "",
        "使用已保存的完整真实轨迹评估9组近距离差分奖励，分析它们对既有回报、量级及排序的影响。所有终止事件沿用原轨迹记录。", "",
        r"\[r_t=-0.01d_{t+1}\,dt-0.2\,dt+500I_{capture,t}-100I_{failure,t}\]", "",
        "代码中的原奖励使用本步积分完成后的距离。这里统一将一次转移写为状态t到t+1，因此原式距离记作d_{t+1}。", "",
        r"\[r_{near,t}=\beta(e^{-d_{t+1}/\lambda}-e^{-d_t/\lambda}),\qquad r'_t=r_t+r_{near,t}\]", "",
        r"\[R=\sum_{t=0}^{T-1}r_t,\qquad G_0=\sum_{t=0}^{T-1}0.99^t r_t\]", "",
        "dt和原奖励系数读取每个Seed的冻结环境配置，gamma读取正式实验快照。G0为完整回合起点的Monte Carlo折扣回报。PPO训练中的GAE仍有其独立计算定义。", "",
        "## 样本与完整性", "",
        "主分析包含5个Seed各500条Best测试轨迹，共2500条，使用相同500个真实随机来袭角度。辅助分析包含各Seed每个唯一Actor一次的固定0°轨迹，共620条。两组各自统计。", "",
        "| 类型 | 主测试样本 | 辅助0°样本 | 分类标准 |",
        "| --- | ---: | ---: | --- |"]
    definitions = dict(capture="原终止标志为捕获", near_miss="失败且5≤最小距离≤10m",
        moderate_failure="失败且10<最小距离≤30m", other_failure="失败且30<最小距离≤100m",
        clear_failure="失败且最小距离>100m")
    for name in CATEGORIES:
        lines.append("| %s | %d | %d | %s |" % (labels[name], manifest["category_counts"][COHORTS[0]][name],
            manifest["category_counts"][COHORTS[1]][name], definitions[name]))
    lines.extend(["", "主测试中全部2244条失败均为防御失败，超时0。Capture覆盖4个Seed，明显失败覆盖2个Seed，其余类别覆盖情况见完整来源清单。辅助轨迹按照Actor哈希去重，177个保存文件中的别名共享一次轨迹。", "",
        "每条轨迹用三维位置独立重算距离和原奖励，逐步最大误差为%.3g。全部源NPZ、索引及配置的SHA256在分析前后匹配，全部回合包含初始和终止帧。" % manifest["original_reward_max_step_error"], "",
        "[来源与逐回合元数据](trajectory_manifest.csv) · [完整重放明细](episode_reward_replay.csv) · [完整性记录](replay_manifest.json)", "",
        "## 原回报与9组参数比较", "",
        "下表使用主2500条轨迹的类别均值。30–50分目标按成功回合的整回合净shaping检查。", "",
        "| λ/m | β | Capture新R | Near miss新R | 中等失败新R | 明显失败新R | Capture净shaping均值 | Capture落在30–50比例 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    baseline = [float(chosen_groups[name]["original_return_mean"]) for name in ORDERED_CATEGORIES]
    lines.append("| 原奖励 | — | %.3f | %.3f | %.3f | %.3f | 0 | — |" % tuple(baseline))
    for row in primary:
        values = [float(row[name + "_new_mean"]) for name in ORDERED_CATEGORIES]
        lines.append("| %g | %g | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f%% |" % (
            float(row["lambda_m"]), float(row["beta"]), *values, float(row["capture_near_mean"]),
            100*float(row["capture_near_target_fraction"])))
    lines.extend(["", "### gamma=0.99的折扣回报", "",
        "| λ/m | β | Capture新G0 | Near miss新G0 | 中等失败新G0 | 明显失败新G0 | 主样本四类完整排序 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    baseline = [float(chosen_groups[name]["original_discounted_return_mean"]) for name in ORDERED_CATEGORIES]
    lines.append("| 原奖励 | — | %.3f | %.3f | %.3f | %.3f | 未满足 |" % tuple(baseline))
    for row in primary:
        values = [float(row[name + "_discounted_new_mean"]) for name in ORDERED_CATEGORIES]
        lines.append("| %g | %g | %.3f | %.3f | %.3f | %.3f | 未满足 |" % (
            float(row["lambda_m"]), float(row["beta"]), *values))
    lines.extend(["", "主样本9组在未折扣和折扣均值上，均为Capture > 中等失败 > Near miss > 明显失败。期望的Near miss > 中等失败仍未实现。辅助0°组的9组未折扣均值达到期望四类排序，折扣均值仍全部未达到。", "",
        "类别均值排序与每一对轨迹的排序具有不同含义。全部相邻类别的逐对胜率，以及逐Seed和逐回合结果可分别从对应CSV核对。", "",
        "[9组完整比较表](parameter_comparison.csv) · [类别统计](category_return_statistics.csv) · [逐对排序](pairwise_ranking.csv)", "",
        "![9组回报均值](plots/nine_parameter_return_comparison.png)", "",
        "## 关键机制核查", "",
        r"令\(\phi_t=e^{-d_t/\lambda}\)，则整回合净贡献严格为", "",
        r"\[S_{near}=\beta(\phi_T-\phi_0)\]", "",
        "该量由终点距离决定。途中接近获得的正项会被后续远离的负项抵消。主样本267条Near miss终点距离为152.862–337.219m，平均192.371m。", "",
        "λ=20m、β=40时，Near miss平均先获得28.2936分正项，随后扣回28.2834分，整回合净加分只有0.0101。中等失败净加分0.0071，原有约0.3566分的类别均值差距仍然存在。", "",
        "折扣后满足", "",
        r"\[S_{near,\gamma}=\beta\left[-\phi_0+(1-\gamma)\sum_{t=1}^{T-1}\gamma^{t-1}\phi_t+\gamma^{T-1}\phi_T\right]\]", "",
        "因此接近后较晚远离会留下正折扣贡献。恒定距离的每步shaping严格为0，闭环往返的未折扣净贡献为0。先正后负的奖励经折扣后通常仍为正。", "",
        "单步符号与整回合望远镜恒等式已经在全部重放中核查。专项单元测试另覆盖等距、往返、所有9组参数和折扣恒等式。", "",
        "## 候选推荐及验收结果", "",
        "推荐保守候选λ=20m、β=40，用于后续讨论。该组合尚未满足本轮完整排序目标。", "",
        "- 主样本256条Capture净加分31.153–35.123，平均32.422，全部落在30–50范围。理论上任意dT<5m的捕获轨迹，净贡献约31.15–40。",
        "- 主样本2244条防御失败最大净加分2.739，最高新总回报−175.488。辅助0°失败最高新总回报−156.707。",
        "- 主样本Capture最低新总回报456.404，最低新G0为14.614，均高于该组全部失败。辅助组Capture与失败的完整分离也保留。",
        "- Capture平均G0由17.447升至21.196，Near miss由−43.080升至−42.605，中等失败由−42.687升至−42.338。Near miss与中等失败的排序差异仍然存在。",
        "- λ=30m、β=40也满足Capture量级，但主样本失败最大净加分6.695。λ=20m对较远距离的影响更小。",
        "- λ=10m、β=60在本批成功样本上也全部达到30–50，但更接近敌机时潜在净贡献可超过50。λ=20m、β=40的终端量级上限更稳健。", "",
        "| 检查要求 | 观察结果 |",
        "| --- | --- |",
        "| Capture > Near miss > 中等失败 > 明显失败 | 主样本9组均未满足，差异集中在Near miss与中等失败 |",
        "| Capture净shaping约30–50 | λ20β40全部满足 |",
        "| 防御失败保持低回报 | 全9组、两组样本均保持负的未折扣总回报 |",
        "| +500/−100主导关系 | 原事件项按记录保留，全9组均保留Capture高于失败的完整样本分离 |",
        "| 擦肩后立即变负 | 全部距离增加的转移满足非正，代表Near miss下一帧为负 |",
        "| 等距时持续加分 | 等距单步严格为0，专项测试通过 |", "",
        "距离差分同时奖励敌机朝我方靠近。辅助Seed1001的0步初始策略，原R=415.252，新R=448.049，原G0=−8.928，新G0=−6.772。这种近基地被动相遇也会获得shaping，须在主动拦截行为解释中保留该限制。", "",
        "本轮按用户给定的φnext−φcurrent原式计算。γ=0.99时，该式与γφnext−φcurrent存在数学差异，折扣后的路径偏好已在上式和逐回合数据中体现。离线重放评估既有轨迹的计分变化，策略更新效果仍需要独立实验验证。", "",
        "## 代表轨迹与逐时间分量", "",
        "每个目标类别从主样本选两条，优先接近该类别最小距离中位数，同时优先不同Seed及相隔至少5°的来袭角。共8条，各图保留全部9组参数。", "",
        "| 类型 | Seed | 角度/° | 最近距离/m | 原R | 新R，λ20β40 | 正项合计 | 负项合计 | 净shaping | 分量图 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for representative in representatives:
        row = chosen[representative["episode_id"]]
        values = [float(row[name]) for name in ("original_return", "new_return", "near_positive_sum", "near_negative_sum", "near_return")]
        lines.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.4f | %.4f | %.6g | [打开](%s) |" % (
            labels[representative["category"]], representative["training_seed"], float(representative["angle_degrees"]),
            float(representative["min_distance"]), *values, representative["component_plot"]))
    lines.extend(["", "分量图依次展示距离、原奖励各分量、9组单步near项和9组累计near项。终止奖励所在面板采用symlog纵轴，正负小项和终止脉冲均可见。全部曲线为真实逐步数据，采用原始采样点。", "",
        "![Near miss真实分量](plots/representative_03_near_miss.png)", "",
        "[代表样本清单](representatives.csv) · [代表样本全部逐步分量CSV](representative_step_components.csv)", "",
        "## 文件与复现", "",
        "新增独立脚本 `scripts/offline_near_reward_replay.py`、分析配置 `configs/offline_near_reward_replay.yaml` 和专项测试 `tests/test_offline_near_reward_replay.py`。运行参数和完整源文件哈希保存在本目录。", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m pytest tests/test_offline_near_reward_replay.py -q",
        ".\\.venv\\Scripts\\python.exe -m scripts.offline_near_reward_replay --config configs/offline_near_reward_replay.yaml", "```", "",
        "重复执行时应在分析配置中指定新的output_dir，入口会拒绝覆盖已有目录。专项测试56 passed，全部3120条真实轨迹原奖励与28,080条候选重放完成数值核查。", ""])
    (output / "OFFLINE_NEAR_REWARD_REPLAY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "README.md").write_text("# 离线近距离奖励重放\n\n"
        "[实验报告](OFFLINE_NEAR_REWARD_REPLAY_REPORT.md)\n\n"
        "- [9组对比表](parameter_comparison.csv)\n"
        "- [完整逐回合重放](episode_reward_replay.csv)\n"
        "- [来源清单](trajectory_manifest.csv)\n"
        "- [逐时间奖励分量图](plots/)\n"
        "- [代表轨迹逐步CSV](representative_step_components.csv)\n"
        "- [源文件完整性核验](replay_manifest.json)\n", encoding="utf-8")


def run(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError("Use a new analysis output directory " + str(output))
    hashes = {str(config_path): sha256(config_path)}
    episodes, gamma = load_sources(config, hashes)
    rows = replay(episodes, config["length_scales_m"], config["betas"], gamma, config["target_capture_shaping_range"])
    comparisons, groups, rankings = summarize(rows, config["length_scales_m"], config["betas"])
    representatives = select_representatives(episodes, config["representatives_per_category"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    source_fields = [name for name in episodes[0] if name not in ("distances", "time", "components")]
    write_csv(output / "trajectory_manifest.csv", [{name: episode[name] for name in source_fields} for episode in episodes])
    write_csv(output / "episode_reward_replay.csv", rows)
    write_csv(output / "parameter_comparison.csv", comparisons)
    write_csv(output / "category_return_statistics.csv", groups)
    write_csv(output / "pairwise_ranking.csv", rankings)
    plot_results(episodes, representatives, rows, comparisons, config, output)
    write_csv(output / "representatives.csv", [{name: episode[name] for name in source_fields + ["component_plot"]} for episode in representatives])
    counts = {cohort: {name: sum(e["cohort"] == cohort and e["category"] == name for e in episodes) for name in CATEGORIES} for cohort in COHORTS}
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise RuntimeError("Source changed during offline analysis " + path)
    manifest = dict(status="completed", gamma=gamma, episode_count=len(episodes), parameter_count=len(config["length_scales_m"]) * len(config["betas"]),
        replay_rows=len(rows), category_counts=counts, source_sha256=hashes,
        original_reward_max_step_error=max(e["max_original_reward_error"] for e in episodes),
        all_source_hashes_unchanged=True,
        history_deduplication="One trajectory per actor_sha256 within each seed, first actual saved occurrence",
        distance_indexing="distances[0]=reset; original reward uses distances[1:]; near=beta*diff(exp(-distances/lambda))",
        discounted_return_definition="sum(gamma**t * reward[t]), t=0..T-1",
        representative_selection="Two nearest category-median min-distance episodes, preferring distinct seeds and angles separated by at least 5 degrees",
        output_dir=str(output.resolve()))
    write_json(output / "replay_manifest.json", manifest)
    write_report(output, manifest)
    print(json.dumps({key:manifest[key] for key in ("status","episode_count","replay_rows","gamma","category_counts","original_reward_max_step_error","output_dir")},ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/offline_near_reward_replay.yaml")
    arguments = parser.parse_args()
    run(arguments.config)
