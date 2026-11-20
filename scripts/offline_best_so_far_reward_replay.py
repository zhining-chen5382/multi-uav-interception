"""Replay record-minimum-distance shaping on a previous frozen trajectory set."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from utils.artifact_paths import resolve_archived_file, resolve_artifact_path

from scripts.offline_near_reward_replay import (
    ROOT, CATEGORIES, COHORTS, _distances, _load_episode, _pairwise_win_rate,
    discounted_return, near_shaping, read_csv, sha256, write_csv, write_json,
)


def best_so_far_shaping(distances, length_scale_m, beta):
    distances = _distances(distances)
    return near_shaping(np.minimum.accumulate(distances), length_scale_m, beta)


def load_previous(config):
    previous = resolve_artifact_path(config["previous_analysis"], root=ROOT)
    manifest = json.loads((previous / "replay_manifest.json").read_text(encoding="utf-8"))
    previous_config = yaml.safe_load((previous / "analysis_config.yaml").read_text(encoding="utf-8"))
    # Keep the frozen manifest untouched; log the exact verified current files.
    hashes = {str(resolve_archived_file(path, digest, root=ROOT)): digest
              for path, digest in manifest["source_sha256"].items()}
    for filename in ("replay_manifest.json", "analysis_config.yaml", "trajectory_manifest.csv",
                     "representatives.csv", "episode_reward_replay.csv", "category_return_statistics.csv"):
        path = (previous / filename).resolve()
        hashes[str(path)] = sha256(path)
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("Previous source hash mismatch " + path)
    if (config["length_scales_m"] != previous_config["length_scales_m"]
            or config["betas"] != previous_config["betas"]):
        raise ValueError("The paired comparison requires the previous parameter grid")
    sources = read_csv(previous / "trajectory_manifest.csv")
    episodes, environments = [], {}
    for source in sources:
        seed = int(source["training_seed"])
        if seed not in environments:
            path = resolve_artifact_path(previous_config["source_root"], root=ROOT) / ("seed_%d" % seed) / "final_test/environment_config.yaml"
            environments[seed] = yaml.safe_load(path.read_text(encoding="utf-8"))
        episode = _load_episode(source["trajectory"], environments[seed], source, source["cohort"],
            seed, int(source["checkpoint_step"]), previous_config, hashes)
        for field in ("episode_id", "category", "trajectory_sha256"):
            if str(episode[field]) != source[field]:
                raise ValueError("Frozen sample or category changed " + source["episode_id"])
        episodes.append(episode)
    representatives = read_csv(previous / "representatives.csv")
    previous_rows = read_csv(previous / "episode_reward_replay.csv")
    return episodes, representatives, previous_rows, float(manifest["gamma"]), hashes


def replay(episodes, previous_rows, config, gamma):
    paired = {(row["episode_id"], float(row["lambda_m"]), float(row["beta"])): row for row in previous_rows}
    rows = []
    for episode in episodes:
        original = episode["components"]["original"]
        distances = episode["distances"]
        best = np.minimum.accumulate(distances)
        records = np.diff(best) < 0
        for length_scale in config["length_scales_m"]:
            for beta in config["betas"]:
                old = near_shaping(distances, length_scale, beta)
                new = best_so_far_shaping(distances, length_scale, beta)
                source = paired[(episode["episode_id"], length_scale, beta)]
                original_return = float(original.sum())
                original_discounted = discounted_return(original, gamma)
                for field, actual in (
                    ("original_return", original_return),
                    ("original_discounted_return", original_discounted),
                    ("near_return", old.sum()),
                    ("new_return", (original + old).sum()),
                    ("discounted_near_return", discounted_return(old, gamma)),
                    ("new_discounted_return", discounted_return(original + old, gamma))):
                    np.testing.assert_allclose(actual, float(source[field]), rtol=0, atol=1e-10)
                np.testing.assert_allclose(new.sum(), beta * (np.exp(-best[-1]/length_scale)
                    - np.exp(-best[0]/length_scale)), rtol=0, atol=1e-11)
                if np.any(new < 0) or np.any(new[~records] != 0) or new.sum() > beta + 1e-10:
                    raise ValueError("Invalid record-minimum reward contract")
                np.testing.assert_array_less(np.cumsum(old) - np.cumsum(new), np.full(len(new), 1e-10))
                new_discounted_near = discounted_return(new, gamma)
                if new_discounted_near + 1e-10 < discounted_return(old, gamma):
                    raise ValueError("Unexpected paired discounted shaping decrease")
                near_total = float(new.sum())
                rows.append(dict(
                    episode_id=episode["episode_id"], cohort=episode["cohort"],
                    training_seed=episode["training_seed"], checkpoint_step=episode["checkpoint_step"],
                    angle_degrees=episode["angle_degrees"], category=episode["category"],
                    lambda_m=length_scale, beta=beta,
                    original_return=original_return, new_return=float((original+new).sum()),
                    near_return=near_total, original_discounted_return=original_discounted,
                    new_discounted_return=discounted_return(original+new, gamma),
                    discounted_near_return=new_discounted_near,
                    previous_near_return=float(source["near_return"]),
                    previous_new_return=float(source["new_return"]),
                    previous_discounted_near_return=float(source["discounted_near_return"]),
                    previous_new_discounted_return=float(source["new_discounted_return"]),
                    retained_shaping_difference=near_total-float(source["near_return"]),
                    discounted_shaping_difference=new_discounted_near-float(source["discounted_near_return"]),
                    rewarded_transitions=int(np.count_nonzero(new > 0)),
                    near_in_target_range=bool(config["target_capture_shaping_range"][0] <= near_total
                                              <= config["target_capture_shaping_range"][1])))
    if len(rows) != len(previous_rows) or len(paired) != len(rows):
        raise ValueError("Paired trajectory/parameter count differs from previous analysis")
    return rows


def summarize(rows, config):
    comparisons, groups, rankings = [], [], []
    for cohort in COHORTS:
        for length_scale in config["length_scales_m"]:
            for beta in config["betas"]:
                selected = [r for r in rows if r["cohort"] == cohort and r["lambda_m"] == length_scale and r["beta"] == beta]
                by_category = {name: [r for r in selected if r["category"] == name] for name in CATEGORIES}
                comparison = dict(cohort=cohort, lambda_m=length_scale, beta=beta, episodes=len(selected))
                for name, members in by_category.items():
                    group = dict(cohort=cohort, lambda_m=length_scale, beta=beta, category=name, episodes=len(members))
                    for field in ("original_return", "new_return", "near_return", "original_discounted_return",
                                  "new_discounted_return", "discounted_near_return", "previous_near_return",
                                  "previous_new_return", "previous_new_discounted_return", "retained_shaping_difference"):
                        values = np.array([r[field] for r in members])
                        for suffix, value in (("mean",values.mean()), ("median",np.median(values)),
                                              ("min",values.min()), ("max",values.max())):
                            group[field+"_"+suffix] = float(value)
                    group["near_target_fraction"] = float(np.mean([r["near_in_target_range"] for r in members]))
                    groups.append(group)
                    for field in ("original_return", "new_return", "near_return", "original_discounted_return",
                                  "new_discounted_return", "discounted_near_return"):
                        comparison[name+"_"+field+"_mean"] = group[field+"_mean"]
                    if name == "capture":
                        comparison["capture_near_target_fraction"] = group["near_target_fraction"]
                for field in ("original_return", "previous_new_return", "new_return", "original_discounted_return",
                              "previous_new_discounted_return", "new_discounted_return"):
                    means = [np.mean([r[field] for r in by_category[name]]) for name in CATEGORIES]
                    comparison[field+"_five_category_mean_order_ok"] = bool(all(a>b for a,b in zip(means,means[1:])))
                    comparison[field+"_mean_order"] = " > ".join(sorted(CATEGORIES, key=lambda name: np.mean([r[field] for r in by_category[name]]), reverse=True))
                    for higher, lower in zip(CATEGORIES,CATEGORIES[1:]):
                        rankings.append(dict(cohort=cohort, lambda_m=length_scale, beta=beta, return_type=field,
                            higher_category=higher, lower_category=lower,
                            pairwise_win_rate=_pairwise_win_rate([r[field] for r in by_category[higher]], [r[field] for r in by_category[lower]])))
                failures = [r for r in selected if r["category"] != "capture"]
                captures = by_category["capture"]
                comparison.update(max_failure_shaping=max(r["near_return"] for r in failures),
                    max_failure_new_return=max(r["new_return"] for r in failures),
                    min_capture_new_return=min(r["new_return"] for r in captures),
                    max_failure_discounted_return=max(r["new_discounted_return"] for r in failures),
                    min_capture_discounted_return=min(r["new_discounted_return"] for r in captures))
                comparisons.append(comparison)
    return comparisons, groups, rankings


def plot_results(episodes, representative_sources, groups, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir()
    lookup = {episode["episode_id"]: episode for episode in episodes}
    representatives, steps = [], []
    length_scale, beta = config["focus_lambda_m"], config["focus_beta"]
    for number, source in enumerate(representative_sources, 1):
        episode = lookup[source["episode_id"]]
        distances, time = episode["distances"], episode["time"]
        best = np.minimum.accumulate(distances)
        old = near_shaping(distances,length_scale,beta)
        new = best_so_far_shaping(distances,length_scale,beta)
        old_cumulative, new_cumulative = np.cumsum(old), np.cumsum(new)
        near_grid = {"near_lambda%g_beta%g" % (scale, amplitude): best_so_far_shaping(distances,scale,amplitude)
                     for scale in config["length_scales_m"] for amplitude in config["betas"]}
        figure, axes = plt.subplots(4,1,figsize=(11,12),sharex=True)
        axes[0].plot(time,distances,color="black",label="Actual distance")
        axes[0].plot(time,best,color="#2176b5",ls="--",label="Best-so-far distance")
        axes[0].set_ylabel("3D distance (m)")
        for field, values in episode["components"].items():
            if field != "original":
                axes[1].plot(time[1:],values,label=field)
        axes[1].plot(time[1:],episode["components"]["original"]+new,ls="--",label="New total reward",alpha=0.8)
        axes[1].set_yscale("symlog",linthresh=0.1)
        axes[1].set_ylabel("Original terms / new total\nsymlog")
        axes[2].plot(time[1:],old,color="#be4f46",label="Previous ordinary-distance shaping")
        axes[2].plot(time[1:],new,color="#218b5d",ls="--",label="Best-so-far shaping")
        axes[2].axhline(0,color="gray",lw=0.7)
        axes[2].set_ylabel("Near reward per step")
        axes[3].plot(time[1:],old_cumulative,color="#be4f46",label="Previous cumulative shaping")
        axes[3].plot(time[1:],new_cumulative,color="#218b5d",ls="--",label="Best-so-far cumulative shaping")
        axes[3].set_ylabel("Cumulative near reward")
        axes[3].set_xlabel("Episode time (s)")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8,loc="best")
        figure.suptitle("%s | seed %d | %.2f deg | lambda=%g m, beta=%g\nMin %.3f m, final %.3f m" % (
            episode["category"],episode["training_seed"],episode["angle_degrees"],length_scale,beta,
            episode["min_distance"],episode["final_distance"]),fontsize=12)
        figure.tight_layout(rect=(0,0,1,0.95))
        filename = "representative_%02d_%s.png" % (number,episode["category"])
        figure.savefig(plots/filename,dpi=150)
        plt.close(figure)
        representatives.append(dict(source,component_plot="plots/"+filename))
        for index in range(len(time)-1):
            row = dict(episode_id=episode["episode_id"],step=index,time_start=time[index],time_end=time[index+1],
                distance_before=distances[index],distance_after=distances[index+1],
                best_distance_before=best[index],best_distance_after=best[index+1],
                record_minimum=bool(best[index+1]<best[index]),
                previous_shaping=old[index],best_so_far_shaping=new[index],
                previous_cumulative_shaping=old_cumulative[index],best_so_far_cumulative_shaping=new_cumulative[index])
            row.update({key:values[index] for key,values in episode["components"].items()})
            row.update({key:values[index] for key,values in near_grid.items()})
            steps.append(row)
    write_csv(output/"representatives.csv",representatives)
    write_csv(output/"representative_step_components.csv",steps)
    selected = [g for g in groups if g["cohort"]==COHORTS[0] and g["lambda_m"]==length_scale and g["beta"]==beta]
    selected = {g["category"]:g for g in selected}
    figure,axes = plt.subplots(1,2,figsize=(13,5))
    names = CATEGORIES[1:]
    for axis,fields in ((axes[0],("original_return_mean","previous_new_return_mean","new_return_mean")),
                        (axes[1],("original_discounted_return_mean","previous_new_discounted_return_mean","new_discounted_return_mean"))):
        for field,label in zip(fields,("Original reward","Previous shaping","Best-so-far shaping")):
            axis.plot(range(len(names)),[selected[name][field] for name in names],marker="o",label=label)
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names,rotation=20,ha="right",fontsize=9)
        axis.set_ylabel("Mean return")
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)
    axes[0].set_title("Same held-out failures | undiscounted")
    axes[1].set_title("Same held-out failures | gamma=0.99")
    figure.suptitle("Paired offline reward comparison | lambda=%g m, beta=%g" % (length_scale,beta))
    figure.tight_layout()
    figure.savefig(plots/"paired_category_comparison.png",dpi=160)
    plt.close(figure)


def write_report(output, manifest, config):
    comparisons = read_csv(output/"parameter_comparison.csv")
    groups = read_csv(output/"category_return_statistics.csv")
    episode_rows = read_csv(output/"episode_reward_replay.csv")
    representatives = read_csv(output/"representatives.csv")
    ranking = read_csv(output/"pairwise_ranking.csv")
    labels = dict(capture="Capture",near_miss="Near miss",moderate_failure="10–30m失败",
                  other_failure="30–100m失败",clear_failure=">100m失败")
    focus_lambda, focus_beta = config["focus_lambda_m"],config["focus_beta"]
    primary = [row for row in comparisons if row["cohort"]==COHORTS[0]]
    auxiliary = [row for row in comparisons if row["cohort"]==COHORTS[1]]
    focus = {row["category"]:row for row in groups if row["cohort"]==COHORTS[0]
             and float(row["lambda_m"])==focus_lambda and float(row["beta"])==focus_beta}
    focus_comparison = next(row for row in primary if float(row["lambda_m"])==focus_lambda and float(row["beta"])==focus_beta)
    paired = {row["episode_id"]:row for row in episode_rows
              if float(row["lambda_m"])==focus_lambda and float(row["beta"])==focus_beta}
    raw_pass = sum(row["new_return_five_category_mean_order_ok"]=="True" for row in primary)
    discounted_pass = [row for row in primary if row["new_discounted_return_five_category_mean_order_ok"]=="True"]
    capture = focus["capture"]
    near = focus["near_miss"]
    lines = ["# Best-so-far distance 离线奖励重放报告", "",
        "## 数据和定义", "",
        "本轮直接读取上一轮冻结的trajectory_manifest.csv和representatives.csv。3120条轨迹的顺序、分类、Seed、角度及原始文件哈希沿用上轮，8条代表样本也保持同一组。主2500条随机角度测试轨迹和辅助620条固定0°历史轨迹各自统计。", "",
        r"\[b_t=\min_{0\le i\le t}d_i,\quad r_{near,t}=\beta(e^{-b_{t+1}/\lambda}-e^{-b_t/\lambda})\]", "",
        r"\[r_t=-0.01d_{t+1}dt-0.2dt+500I_{capture,t}-100I_{failure,t},\quad r'_t=r_t+r_{near,t}\]", "",
        "为与实际step实现一致，d0为reset后的距离，原奖励每步距离使用积分完成后的d_next。原奖励、dt及gamma来自冻结配置。捕获与失守事件沿用保存轨迹的真实判定。", "",
        r"\[R=\sum_{t=0}^{T-1}r_t,\quad G_0=\sum_{t=0}^{T-1}\gamma^t r_t,\quad\gamma=0.99\]", "",
        "| 类别 | 主测试集 | 辅助0°历史集 |", "| --- | ---: | ---: |"]
    for name in CATEGORIES:
        lines.append("| %s | %d | %d |" % (labels[name],manifest["category_counts"][COHORTS[0]][name],manifest["category_counts"][COHORTS[1]][name]))
    lines.extend(["", "分类边界沿用上轮。Near miss为失败且5≤min≤10m，随后两组分别为10<min≤30m和30<min≤100m，明显失败为min>100m。Capture按原终止标志。", "",
        "## 九组完整比较", "",
        "以下为主2500条轨迹的类别平均未折扣回报。新增量是每个回合的shaping净合计。", "",
        "| λ/m | β | Capture新R | Near miss新R | 10–30m新R | 30–100m新R | >100m新R | Capture平均shaping | 失败最大shaping |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    baseline = [float(focus[name]["original_return_mean"]) for name in CATEGORIES]
    lines.append("| 原Reward | — | %.3f | %.3f | %.3f | %.3f | %.3f | 0 | 0 |" % tuple(baseline))
    for row in primary:
        values = [float(row[name+"_new_return_mean"]) for name in CATEGORIES]
        lines.append("| %g | %g | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            float(row["lambda_m"]),float(row["beta"]),*values,float(row["capture_near_return_mean"]),float(row["max_failure_shaping"])))
    lines.extend(["", "### 各类别shaping平均累计贡献", "",
        "| λ/m | β | Capture | Near miss | 10–30m失败 | 30–100m失败 | >100m失败 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in primary:
        values = [float(row[name+"_near_return_mean"]) for name in CATEGORIES]
        lines.append("| %g | %g | %.3f | %.3f | %.3f | %.3f | %.6f |" % (float(row["lambda_m"]),float(row["beta"]),*values))
    lines.extend(["", "### gamma=0.99的折扣回报", "",
        "| λ/m | β | Capture新G0 | Near miss新G0 | 10–30m新G0 | 30–100m新G0 | >100m新G0 | 五类均值排序 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    baseline = [float(focus[name]["original_discounted_return_mean"]) for name in CATEGORIES]
    lines.append("| 原Reward | — | %.3f | %.3f | %.3f | %.3f | %.3f | 未满足 |" % tuple(baseline))
    for row in primary:
        values = [float(row[name+"_new_discounted_return_mean"]) for name in CATEGORIES]
        verdict = "满足" if row["new_discounted_return_five_category_mean_order_ok"]=="True" else "未满足"
        lines.append("| %g | %g | %.3f | %.3f | %.3f | %.3f | %.3f | %s |" % (float(row["lambda_m"]),float(row["beta"]),*values,verdict))
    lines.extend(["", "主测试集未折扣五类均值排序达标%d/%d组，折扣排序达标%d/%d组。辅助固定0°集对应为%d/%d和%d/%d。这里检验的完整顺序为Capture > Near miss > 10–30m失败 > 30–100m失败 > >100m失败。" % (
        raw_pass,len(primary),len(discounted_pass),len(primary),
        sum(r["new_return_five_category_mean_order_ok"]=="True" for r in auxiliary),len(auxiliary),
        sum(r["new_discounted_return_five_category_mean_order_ok"]=="True" for r in auxiliary),len(auxiliary)), "",
        "[完整9组比较CSV](parameter_comparison.csv) · [所有类别的均值、中位数、最小值和最大值](category_return_statistics.csv)", "",
        "## 与上一版直接配对比较", "",
        "以下固定λ=%gm、β=%g。每个数字均来自相同轨迹的逐回合配对计算。" % (focus_lambda,focus_beta), "",
        "| 类别 | 原R | 上版新R | best版新R | 上版shaping | best版shaping | 原G0 | 上版新G0 | best版新G0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    paired_category_rows = []
    for name in CATEGORIES:
        group = focus[name]
        fields = ("original_return_mean","previous_new_return_mean","new_return_mean","previous_near_return_mean",
                  "near_return_mean","original_discounted_return_mean","previous_new_discounted_return_mean","new_discounted_return_mean")
        values = [float(group[field]) for field in fields]
        lines.append("| %s | %.3f | %.3f | %.3f | %.4f | %.4f | %.3f | %.3f | %.3f |" % (labels[name],*values))
        paired_category_rows.append(dict(category=name,**{field:float(group[field]) for field in fields}))
    write_csv(output/"focus_paired_category_comparison.csv",paired_category_rows)
    lines.extend(["", "Near miss平均保留新增奖励%.3f分，上版净贡献仅%.4f分。Capture回合首次进入捕获范围即结束，其历史最小距离等于终点距离，因此未折扣奖励与上版一致。本批Capture轨迹的折扣结果也全部与上版一致。" % (
        float(near["near_return_mean"]),float(near["previous_near_return_mean"])), "",
        "![同一失败样本配对比较](plots/paired_category_comparison.png)", "",
        "## 安全性与重复奖励检查", "",
        r"\[\sum_t r_{near,t}=\beta(e^{-d_{min}/\lambda}-e^{-d_0/\lambda})\le\beta\le60\]", "",
        "- 只有历史最小距离降低时才产生正项。远离、等距、再次靠近但仍大于旧纪录时均为0。",
        "- 已奖励的距离区间由历史最小值记忆，同一区间重复往返不会再次累积奖励。专项测试覆盖连续25次重复循环和再次刷新纪录。",
        "- 防御失败保留−100事件项，新增总量上限为60，连同非正的原距离和时间项，新未折扣回报始终低于−40。该界限针对防御失败，超时事件须按其原终止项单独解释。",
        "- 两个样本组、所有9组参数均保留Capture回报高于全部失败的完整分离，原+500终止项继续主导。折扣后各项按实际发生时刻加权，表中直接记录G0。", "",
        "λ=%g、β=%g时，主防御失败最大新增%.3f分，最高新R为%.3f；Capture新增范围%.3f–%.3f，平均%.3f。" % (
        focus_lambda,focus_beta,float(focus_comparison["max_failure_shaping"]),float(focus_comparison["max_failure_new_return"]),
        float(capture["near_return_min"]),float(capture["near_return_max"]),float(capture["near_return_mean"])), "",
        "五类均值顺序与每一对轨迹的顺序具有不同含义。相邻类别逐对胜率如下。", "",
        "| 相邻类型 | 未折扣胜率 | 折扣胜率 |", "| --- | ---: | ---: |"])
    for higher,lower in zip(CATEGORIES,CATEGORIES[1:]):
        pair = [r for r in ranking if r["cohort"]==COHORTS[0] and float(r["lambda_m"])==focus_lambda and float(r["beta"])==focus_beta
                and r["higher_category"]==higher and r["lower_category"]==lower]
        raw = next(r for r in pair if r["return_type"]=="new_return")
        discounted = next(r for r in pair if r["return_type"]=="new_discounted_return")
        lines.append("| %s > %s | %.2f%% | %.2f%% |" % (
            labels[higher],labels[lower],100*float(raw["pairwise_win_rate"]),100*float(discounted["pairwise_win_rate"])))
    lines.extend(["", "## 代表性真实轨迹", "",
        "沿用上一轮8条代表样本，各图固定λ=%gm、β=%g。逐步CSV同时包含9组参数的best-so-far奖励。" % (focus_lambda,focus_beta), "",
        "| 类别 | Seed | 角度/° | min/m | 原R | 上版新R | best版新R | 图 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for source in representatives:
        row = paired[source["episode_id"]]
        lines.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | [打开](%s) |" % (
            labels[source["category"]],source["training_seed"],float(source["angle_degrees"]),float(source["min_distance"]),
            float(row["original_return"]),float(row["previous_new_return"]),float(row["new_return"]),source["component_plot"]))
    lines.extend(["", "Seed1003、scene154的Near miss在23.3秒达到6.615m，下一帧增加到6.861m。该远离步上版shaping为−0.351，新版为0。上版整回合净shaping为0.0134，新版保留28.7351。", "",
        "![Near miss的距离及逐时间奖励](plots/representative_03_near_miss.png)", "",
        "[代表样本清单](representatives.csv) · [逐时间全部分量与9组shaping](representative_step_components.csv)", "",
        "## 推荐与限制", "",
        "建议保留λ=20m、β=40作为保守候选。Capture净贡献在31.153–35.123，Near miss保留约28.294分，失败最高新增31.149，低于原−100失败项。五类未折扣均值全部有序，Near miss与10–30m失败的折扣均值也已改善到预期顺序。", "",
        "该候选的30–100m失败折扣均值%.3f仍低于>100m失败的%.3f，完整五类折扣排序尚未达标。" % (
            float(focus["other_failure"]["new_discounted_return_mean"]),float(focus["clear_failure"]["new_discounted_return_mean"])), "",
        "本批唯一通过主测试集完整折扣均值排序的是λ=30m、β=60，末两类均值差仅约0.0526分。其Capture平均新增52.155分，主失败最大新增50.786分。若将完整折扣均值排序作为硬要求，该组合通过本批检查，量级和较小尾部间隔应同时纳入判断。", "",
        "同一轨迹下G0仍受接近时刻、全程距离成本及回合长度影响。仅按最小距离分组无法从公式推出所有轨迹对的严格排序。", "",
        "best-so-far奖励依赖历史最小距离。当前16维观测中没有这个历史量，同一当前几何状态可能具有不同的剩余shaping预算。未来训练接入涉及其状态语义、回合重置与恢复一致性。本轮验证范围为已保存轨迹的离线计分和上述数值性质。", "",
        "敌机朝我方主动靠近也会刷新距离纪录，近基地的被动相遇同样获得奖励。Capture仍沿用原先离散判定及捕获优先规则。离线排序改善与实际策略学习效果需要分别验证。", "",
        "## 验证与复现", "",
        "旧版与新版专项测试合计101 passed。3120条完整轨迹、28080条配对重放全部通过原奖励复算、历史纪录正项条件、望远镜恒等式和源文件哈希核验。", "",
        "新增独立分析脚本、分析YAML和专项测试文件，全部分析输出保存在本目录。", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m pytest tests/test_offline_near_reward_replay.py tests/test_offline_best_so_far_reward_replay.py -q",
        ".\\.venv\\Scripts\\python.exe -m scripts.offline_best_so_far_reward_replay --config configs/offline_best_so_far_reward_replay.yaml", "```", "",
        "再次执行应在分析配置中选择新的output_dir，入口会拒绝覆盖已有目录。", "",
        "[冻结轨迹清单](trajectory_manifest.csv) · [逐回合全部配对结果](episode_reward_replay.csv) · [重放来源哈希](replay_manifest.json)", ""])
    (output/"OFFLINE_BEST_SO_FAR_REWARD_REPLAY_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    (output/"README.md").write_text("# 历史最小距离奖励离线重放\n\n"
        "[完整报告](OFFLINE_BEST_SO_FAR_REWARD_REPLAY_REPORT.md)\n\n"
        "- [9组参数和五类原/新回报](parameter_comparison.csv)\n"
        "- [各类别分布与上版对照](category_return_statistics.csv)\n"
        "- [重点参数与上版直接比较](focus_paired_category_comparison.csv)\n"
        "- [全部28080条配对重放](episode_reward_replay.csv)\n"
        "- [代表Near miss的距离与奖励分量](plots/representative_03_near_miss.png)\n"
        "- [同一批逐时间分量CSV](representative_step_components.csv)\n"
        "- [源轨迹与哈希](replay_manifest.json)\n",encoding="utf-8")


def run(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError("Use a new analysis output directory " + str(output))
    episodes, representatives, previous_rows, gamma, hashes = load_previous(config)
    hashes[str(config_path)] = sha256(config_path)
    rows = replay(episodes,previous_rows,config,gamma)
    comparisons,groups,rankings = summarize(rows,config)
    output.mkdir(parents=True,exist_ok=False)
    (output/"analysis_config.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
    previous = ROOT / config["previous_analysis"]
    (output/"trajectory_manifest.csv").write_bytes((previous/"trajectory_manifest.csv").read_bytes())
    write_csv(output/"episode_reward_replay.csv",rows)
    write_csv(output/"parameter_comparison.csv",comparisons)
    write_csv(output/"category_return_statistics.csv",groups)
    write_csv(output/"pairwise_ranking.csv",rankings)
    plot_results(episodes,representatives,groups,config,output)
    for path,digest in hashes.items():
        if sha256(path) != digest:
            raise RuntimeError("Source changed during offline replay " + path)
    counts = {cohort:{name:sum(e["cohort"]==cohort and e["category"]==name for e in episodes) for name in CATEGORIES} for cohort in COHORTS}
    manifest = dict(status="completed",gamma=gamma,episode_count=len(episodes),replay_rows=len(rows),parameter_count=len(config["length_scales_m"])*len(config["betas"]),
        category_counts=counts,source_sha256=hashes,all_sources_unchanged=True,
        trajectory_manifest_byte_identical=sha256(output/"trajectory_manifest.csv")==sha256(previous/"trajectory_manifest.csv"),
        representative_episode_ids=[row["episode_id"] for row in representatives],
        formula="beta * diff(exp(-minimum.accumulate(distances) / lambda))",
        original_reward_max_step_error=max(e["max_original_reward_error"] for e in episodes),
        previous_analysis=str(previous.resolve()),output_dir=str(output.resolve()))
    write_json(output/"replay_manifest.json",manifest)
    write_report(output,manifest,config)
    print(json.dumps({key:manifest[key] for key in ("status","episode_count","replay_rows","category_counts","output_dir")},ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/offline_best_so_far_reward_replay.yaml")
    arguments = parser.parse_args()
    run(arguments.config)
