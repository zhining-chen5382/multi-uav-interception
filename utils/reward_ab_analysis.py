"""Read-only, matched-budget comparison of original and shaped formal PPO runs."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


DISTANCE_BINS = ("lt5", "5_to_10", "10_to_30", "30_to_100", "gt100")
PPO_FIELDS = ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
              "std_x", "std_y", "std_z", "actor_learning_rate", "critic_learning_rate")


def distance_bin(distance):
    value = float(distance)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Minimum distance must be finite and nonnegative")
    if value < 5:
        return "lt5"
    if value <= 10:
        return "5_to_10"
    if value <= 30:
        return "10_to_30"
    if value <= 100:
        return "30_to_100"
    return "gt100"


def _bool(value):
    return value is True or str(value).lower() == "true"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(path, hashes):
    path = Path(path).resolve()
    hashes[str(path)] = _sha(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path, hashes):
    path = Path(path).resolve()
    hashes[str(path)] = _sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path, hashes):
    path = Path(path).resolve()
    hashes[str(path)] = _sha(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_csv(path, rows):
    if not rows:
        raise ValueError("A comparison table cannot be empty")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, field):
    return float(np.mean([float(row[field]) for row in rows])) if rows else None


def _minimum(row):
    return float(row["min_distance"] if "min_distance" in row else row["minimum_distance"])


def _reward_values(row, group):
    if group == "A":
        return float(row["episode_return"]), 0.0
    if "original_return" not in row or "near_shaping_return" not in row:
        raise ValueError("B episode logs must record original_return and near_shaping_return")
    original, near = float(row["original_return"]), float(row["near_shaping_return"])
    np.testing.assert_allclose(original + near, float(row["episode_return"]), rtol=0, atol=1e-8)
    return original, near


def summarize_training(episodes, validation, summary, group, seed, budget, interval):
    if summary["env_steps"] != budget or summary["status"] != "completed":
        raise ValueError("A/B comparison requires completed runs with exactly the same budget")
    if not episodes or not validation or int(validation[-1]["env_steps"]) != budget:
        raise ValueError("Completed training and final validation records are required")
    expected = list(range(interval, budget + 1, interval))
    if [int(row["env_steps"]) for row in validation] != expected:
        raise ValueError("Validation milestones differ from the fixed protocol")
    if any(int(row["env_steps"]) > budget for row in episodes):
        raise ValueError("Episode records exceed the comparison budget")
    capture_count = sum(_bool(row["captured"]) for row in episodes)
    original, near = zip(*[_reward_values(row, group) for row in episodes])
    best_rate = max(float(row["capture_rate"]) for row in validation)
    best_step = int(summary["best_env_steps"])
    selected = next((row for row in validation if int(row["env_steps"]) == best_step), None)
    if selected is None or float(selected["capture_rate"]) != best_rate:
        raise ValueError("Selected best checkpoint disagrees with validation capture rate")
    positives = [int(row["env_steps"]) for row in validation if float(row["capture_rate"]) > 0]
    streak, longest = 0, 0
    for row in validation:
        streak = streak + 1 if float(row["capture_rate"]) > 0 else 0
        longest = max(longest, streak)
    after_best = [float(row["capture_rate"]) for row in validation if int(row["env_steps"]) > best_step]
    lost_success = sum(float(left["capture_rate"]) > 0 and float(right["capture_rate"]) == 0
                       for left, right in zip(validation, validation[1:]))
    result = dict(group=group, seed=seed, budget=budget, episodes=len(episodes),
        training_captures=capture_count, training_capture_rate=capture_count / len(episodes),
        first_training_capture_step=min((int(row["env_steps"]) for row in episodes if _bool(row["captured"])), default=None),
        first_positive_validation_step=min(positives) if positives else None,
        positive_validations=len(positives), longest_positive_validation_streak=longest,
        best_validation_capture_rate=best_rate, best_checkpoint_step=best_step,
        final_validation_capture_rate=float(validation[-1]["capture_rate"]),
        peak_to_final_drop_pp=100*(best_rate-float(validation[-1]["capture_rate"])),
        post_best_validation_count=len(after_best),
        minimum_capture_rate_after_best=min(after_best) if after_best else None,
        positive_to_zero_validation_transitions=lost_success,
        training_original_return_mean=float(np.mean(original)), training_near_shaping_mean=float(np.mean(near)),
        training_logged_return_mean=_mean(episodes,"episode_return"))
    windows = []
    for end in expected:
        members = [row for row in episodes if end-interval < int(row["env_steps"]) <= end]
        captures = sum(_bool(row["captured"]) for row in members)
        windows.append(dict(group=group,seed=seed,window_start=end-interval,window_end=end,
            completed_episodes=len(members),captures=captures,
            capture_rate=captures/len(members) if members else None,
            near_miss_episodes=sum(distance_bin(_minimum(row))=="5_to_10" for row in members),
            original_return_mean=float(np.mean([_reward_values(row,group)[0] for row in members])) if members else None))
    return result, windows


def _scenario_key(row):
    return int(row["scenario_id"]), int(row["seed"]), float(row["angle_degrees"])


def _verify_shaped_evaluation(row, directory, environment, hashes):
    """Independently reconstruct original and shaped step rewards from saved positions."""
    path = (directory / row["trajectory"]).resolve()
    hashes[str(path)] = _sha(path)
    with np.load(path, allow_pickle=False) as archive:
        states, recorded = archive["states"], archive["rewards"]
    distance = np.linalg.norm(states[:,0,:3] - states[:,1,:3],axis=1)
    coefficient = environment["reward"]
    shaping = coefficient["best_so_far_shaping"]
    near = shaping["beta"] * np.diff(np.exp(-np.minimum.accumulate(distance)/shaping["length_scale_m"]))
    original = -coefficient["distance_cost_coefficient"] * distance[1:] * environment["environment"]["dt"]
    original += coefficient["survival_penalty_per_second"] * environment["environment"]["dt"]
    original[-1] += coefficient["capture_reward"] if _bool(row["captured"]) else 0
    original[-1] += coefficient["attack_failure_penalty"] if _bool(row["defense_failure"]) else 0
    np.testing.assert_allclose(original+near,recorded,rtol=0,atol=1e-9)
    np.testing.assert_allclose((original+near).sum(),float(row["episode_return"]),rtol=0,atol=1e-8)
    if "near_shaping_return" in row:
        np.testing.assert_allclose(near.sum(),float(row["near_shaping_return"]),rtol=0,atol=1e-8)
    if "original_return" in row:
        np.testing.assert_allclose(original.sum(),float(row["original_return"]),rtol=0,atol=1e-8)
    return float(original.sum()), float(near.sum())


def summarize_evaluation(rows, directory, environment, group, seed, model, hashes):
    captures = sum(_bool(row["captured"]) for row in rows)
    bins = Counter(distance_bin(_minimum(row)) for row in rows)
    rewards = [_verify_shaped_evaluation(row,directory,environment,hashes) for row in rows] if group=="B" else [_reward_values(row,group) for row in rows]
    summary = dict(group=group,seed=seed,model=model,episodes=len(rows),captures=captures,
        capture_rate=captures/len(rows),defense_failure_rate=float(np.mean([_bool(r["defense_failure"]) for r in rows])),
        timeout_rate=float(np.mean([_bool(r["timeout"]) for r in rows])),
        mean_minimum_distance=_mean(rows,"min_distance"),mean_episode_length=_mean(rows,"episode_length"),
        original_return_mean=float(np.mean([r[0] for r in rewards])),
        near_shaping_mean=float(np.mean([r[1] for r in rewards])),logged_return_mean=_mean(rows,"episode_return"))
    for name in DISTANCE_BINS:
        summary[name+"_count"] = bins[name]
        summary[name+"_fraction"] = bins[name]/len(rows)
    signs = []
    for sign in ("negative","zero","positive"):
        members = [row for row in rows if ("negative" if float(row["angle_degrees"])<0 else "positive" if float(row["angle_degrees"])>0 else "zero")==sign]
        wins = sum(_bool(row["captured"]) for row in members)
        signs.append(dict(group=group,seed=seed,model=model,angle_sign=sign,episodes=len(members),captures=wins,
                          capture_rate=wins/len(members) if members else None))
    shaping_rows = []
    for name in ("capture",)+DISTANCE_BINS[1:]:
        indices = [index for index,row in enumerate(rows) if ("capture" if _bool(row["captured"]) else distance_bin(_minimum(row)))==name]
        values = [rewards[index][1] for index in indices]
        if values:
            shaping_rows.append(dict(group=group,seed=seed,source="evaluation_"+model,category=name,episodes=len(values),
                near_shaping_mean=float(np.mean(values)),near_shaping_min=float(np.min(values)),near_shaping_max=float(np.max(values))))
    return summary, signs, shaping_rows


def _plots(output, validation, windows, updates, evaluations, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(A="#2e78a6",B="#df7f30")
    figure,axes = plt.subplots(2,len(seeds),figsize=(5*len(seeds),8),squeeze=False)
    for column,seed in enumerate(seeds):
        for group in ("A","B"):
            rows = [r for r in validation if r["seed"]==seed and r["group"]==group]
            axes[0,column].plot([int(r["env_steps"]) for r in rows],[float(r["capture_rate"])*100 for r in rows],label=group,color=colors[group],marker=".")
            rows = [r for r in windows if r["seed"]==seed and r["group"]==group and r["capture_rate"] is not None]
            axes[1,column].plot([r["window_end"] for r in rows],[r["capture_rate"]*100 for r in rows],label=group,color=colors[group])
        axes[0,column].set_title("Seed %d" % seed)
        axes[0,column].set_ylabel("Validation capture (%)")
        axes[1,column].set_ylabel("Training capture (%) / 10k window")
        for axis in axes[:,column]:
            axis.set_xlabel("Aggregate environment steps")
            axis.grid(alpha=.2)
            axis.legend()
    figure.tight_layout()
    figure.savefig(output/"capture_learning_curves.png",dpi=160)
    plt.close(figure)
    figure,axes = plt.subplots(4,len(seeds),figsize=(5*len(seeds),13),squeeze=False)
    for column,seed in enumerate(seeds):
        for group in ("A","B"):
            rows = [r for r in updates if r["seed"]==seed and r["group"]==group]
            steps = [int(r["env_steps"]) for r in rows]
            for dimension,style in zip("xyz",("-","--",":")):
                axes[0,column].plot(steps,[float(r["std_"+dimension]) for r in rows],color=colors[group],ls=style,label=group+" std_"+dimension)
            for index,field in enumerate(("entropy","approx_kl","clip_fraction"),1):
                axes[index,column].plot(steps,[float(r[field]) for r in rows],color=colors[group],label=group)
        axes[0,column].set_title("Seed %d" % seed)
        for index,label in enumerate(("Action std","Entropy","KL","Clip fraction")):
            axes[index,column].set_ylabel(label)
            axes[index,column].grid(alpha=.2)
            axes[index,column].legend(fontsize=8)
        axes[-1,column].set_xlabel("Aggregate environment steps")
    figure.tight_layout()
    figure.savefig(output/"ppo_diagnostics.png",dpi=160)
    plt.close(figure)
    figure,axes = plt.subplots(1,2,figsize=(12,4.5))
    x = np.arange(len(DISTANCE_BINS))
    for column,model in enumerate(("best","final")):
        for offset,group in ((-.18,"A"),(.18,"B")):
            rows = [r for r in evaluations if r["group"]==group and r["model"]==model]
            counts = [sum(r[name+"_count"] for r in rows) for name in DISTANCE_BINS]
            axes[column].bar(x+offset,counts,width=.36,label=group,color=colors[group])
        axes[column].set_xticks(x)
        axes[column].set_xticklabels(("<5m","5-10m","10-30m","30-100m",">100m"))
        axes[column].set_title(model+" checkpoints | pooled matched test scenes")
        axes[column].set_ylabel("Episodes")
        axes[column].legend()
    figure.tight_layout()
    figure.savefig(output/"test_distance_distribution.png",dpi=160)
    plt.close(figure)


def _report(output,training,evaluations,signs,shaping,seeds,budget,metadata):
    lines = ["# Original reward vs best-so-far reward A/B报告","",
        "## 对照范围","",
        "A组复用原正式实验的相同Seed、累计%d步结果。B组从零训练，新增best-so-far shaping，其余训练与验证设置沿用冻结配置。两个定义的总Reward分别记录，共同original_return用于同尺度对照。" % budget,"",
        "训练捕获率按已完成回合统计。未完成回合仍计入aggregate步数，回合捕获率分母使用日志中的已完成回合。训练窗口按原验证间隔划分。","",
        "## 每Seed训练与验证","",
        "| Group | Seed | 训练捕获 | 训练捕获率 | 首次训练捕获步数 | 首次正验证步数 | Best验证捕获率 | Best步数 | Final验证捕获率 | Best→Final下降/百分点 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in training:
        lines.append("| %s | %d | %d/%d | %.3f%% | %s | %s | %.1f%% | %d | %.1f%% | %.1f |" % (
            row["group"],row["seed"],row["training_captures"],row["episodes"],100*row["training_capture_rate"],
            row["first_training_capture_step"],row["first_positive_validation_step"],100*row["best_validation_capture_rate"],
            row["best_checkpoint_step"],100*row["final_validation_capture_rate"],row["peak_to_final_drop_pp"]))
    pooled = {}
    for group in ("A","B"):
        rows = [r for r in training if r["group"]==group]
        count,episodes = sum(r["training_captures"] for r in rows),sum(r["episodes"] for r in rows)
        pooled[group] = count/episodes
        lines.extend(["","%s组合计训练捕获%d/%d，捕获率%.4f%%。" % (group,count,episodes,100*count/episodes)])
    changes = [next(r for r in training if r["group"]=="B" and r["seed"]==seed)["training_capture_rate"]
               - next(r for r in training if r["group"]=="A" and r["seed"]==seed)["training_capture_rate"] for seed in seeds]
    lines.extend(["","B−A合并训练捕获率差为%.4f个百分点，逐Seed改善%d/%d。历史约0.56%%来自另一统计范围，本报告使用上述匹配预算的A组重新计算值。" % (100*(pooled["B"]-pooled["A"]),sum(v>0 for v in changes),len(seeds)),"",
        "![捕获学习曲线](capture_learning_curves.png)","",
        "## Best和Final的相同场景测试","",
        "| Group | Seed | 模型 | 捕获率 | 5–10m占比 | 最小距离均值/m | 共同原Reward均值 | 新增shaping均值 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in evaluations:
        lines.append("| %s | %d | %s | %.2f%% | %.2f%% | %.3f | %.3f | %.3f |" % (row["group"],row["seed"],row["model"],100*row["capture_rate"],100*row["5_to_10_fraction"],row["mean_minimum_distance"],row["original_return_mean"],row["near_shaping_mean"]))
    lines.extend(["","![测试距离分布](test_distance_distribution.png)","",
        "## 正负方向测试","","| Group | Seed | 模型 | 方向 | 捕获/回合 | 捕获率 |","| --- | ---: | --- | --- | ---: | ---: |"])
    for row in signs:
        if row["episodes"]:
            lines.append("| %s | %d | %s | %s | %d/%d | %.2f%% |" % (row["group"],row["seed"],row["model"],row["angle_sign"],row["captures"],row["episodes"],100*row["capture_rate"]))
    lines.extend(["","## PPO更新与策略保持","",
        "std、entropy、KL和clip fraction的原始更新序列全部保留。Best出现时刻、后续最差捕获率、Final下降幅度和正验证最长连续次数见training_summary.csv。曲线使用真实数据，无平滑。","",
        "![PPO诊断](ppo_diagnostics.png)","",
        "### 保持性观察范围", "",
        "最佳点之后的观察严格排除最佳点自身。后续验证次数为0时，保持性尚无后续观察。中途成功后回到0也单独计数。", "",
        "| Group | Seed | 正验证次数 | 最长连续正验证 | 成功→0次数 | Best后验证次数 | Best后最低捕获率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in training:
        minimum = row["minimum_capture_rate_after_best"]
        observed = "尚无后续观察" if minimum is None else "%.1f%%" % (100*minimum)
        lines.append("| %s | %d | %d | %d | %d | %d | %s |" % (
            row["group"], row["seed"], row["positive_validations"], row["longest_positive_validation_streak"],
            row["positive_to_zero_validation_transitions"], row["post_best_validation_count"], observed))
    lines.extend(["",
        "## 奖励量级","",
        "训练B组直接使用环境累计near_shaping_return。测试B组另外从每条NPZ位置轨迹重算原奖励和best-so-far逐步奖励，并与在线保存rewards及终止指标核对。Capture、Near miss、各失败距离类别的累计量级见shaping_by_category.csv。","",
        "best-so-far项对敌机主动接近同样产生奖励。历史最小距离属于环境状态，Actor/Critic仍采用原16维观测。","",
        "## 解释范围","",
        "三个Seed全部报告。三次运行提供本轮受控对照，捕获率差和连续验证结果用于描述改善与退化，统计显著性及更广泛泛化尚需更多独立运行。后续五Seed实验应另行决策。","",
        "A、B使用各自300k范围内选择的Best。相同500个测试场景按scenario_id、环境seed及来袭角度逐项配对，测试结果与固定0°视频单例分别解释。","",
        "## 文件与来源","",
        "- [逐Seed训练与验证汇总](training_summary.csv)","- [每10k训练窗口](training_windows.csv)",
        "- [每10k验证原始数据](validation_history.csv)","- [PPO原始更新](ppo_updates.csv)",
        "- [PPO数值摘要](ppo_summary.csv)","- [Best/Final完整测试指标](test_summary.csv)",
        "- [方向捕获率](angle_sign_capture.csv)","- [新增奖励类别分布](shaping_by_category.csv)",
        "- [源文件及版本](comparison_manifest.json)","",
        "代码版本按每个Run原metadata记录，原源文件SHA256在分析前后核对。A/B报告仅写入新的comparison目录。",""
    ])
    path = output/"REWARD_AB_REPORT.md"
    path.write_text("\n".join(lines),encoding="utf-8")
    return path


def analyze_experiment(output_root, baseline_root):
    """Analyze all configured seeds after both models' paired tests complete."""
    output_root,baseline_root = Path(output_root).resolve(),Path(baseline_root).resolve()
    candidate = output_root/"comparison"
    if (candidate/"comparison_manifest.json").is_file():
        old = json.loads((candidate/"comparison_manifest.json").read_text(encoding="utf-8"))
        if (old.get("analysis_implementation_sha256")==_sha(Path(__file__))
                and old["baseline_root"]==str(baseline_root)
                and all(Path(p).is_file() and _sha(p)==h for p,h in old["source_sha256"].items())):
            return candidate/"REWARD_AB_REPORT.md"
    counter=2
    while candidate.exists():
        candidate=output_root/("comparison_%03d" % counter)
        counter+=1
    hashes={}
    config=_yaml(output_root/"experiment_config.yaml",hashes)
    seeds=config["experiment"]["seeds"]
    budget=config["training"]["total_env_steps"]
    interval=config["training"]["evaluation_interval_steps"]
    training,windows,validations,updates,evaluations,signs,shaping,ppo,metadata=[],[],[],[],[],[],[],[],[]
    reference_scenarios=None
    for seed in seeds:
        for group,root in (("A",baseline_root),("B",output_root)):
            run=root/("seed_%d" % seed)
            summary=_json(run/"summary.json",hashes)
            environment=_yaml(run/"environment_config.yaml",hashes)
            episodes=_csv(run/"episodes.csv",hashes)
            validation=_csv(run/"validation.csv",hashes)
            metrics,window=summarize_training(episodes,validation,summary,group,seed,budget,interval)
            training.append(metrics)
            windows.extend(window)
            validations.extend(dict(row,group=group,seed=seed) for row in validation)
            new_updates=_csv(run/"updates.csv",hashes)
            updates.extend(dict(row,group=group,seed=seed) for row in new_updates)
            numeric=dict(group=group,seed=seed)
            for field in PPO_FIELDS:
                values=np.asarray([float(row[field]) for row in new_updates])
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite PPO diagnostic "+field)
                numeric.update({field+"_mean":float(values.mean()),field+"_min":float(values.min()),
                                field+"_max":float(values.max()),field+"_last":float(values[-1])})
            ppo.append(numeric)
            if (run/"metadata.json").exists():
                meta=_json(run/"metadata.json",hashes)
                metadata.append(dict(group=group,seed=seed,code_version=meta.get("code_version"),git_dirty=meta.get("git_dirty")))
            for name in ("capture",)+DISTANCE_BINS[1:]:
                members=[row for row in episodes if ("capture" if _bool(row["captured"]) else distance_bin(_minimum(row)))==name]
                if members:
                    values=[_reward_values(row,group)[1] for row in members]
                    shaping.append(dict(group=group,seed=seed,source="training",category=name,episodes=len(values),
                        near_shaping_mean=float(np.mean(values)),near_shaping_min=float(np.min(values)),near_shaping_max=float(np.max(values))))
            for model in ("best","final"):
                directory=run/("final_test" if model=="best" else "final_model_test")
                if group=="A" and model=="final" and not (directory/"episodes.csv").exists():
                    directory=output_root/"baseline_evaluation"/("seed_%d" % seed)/"final"
                rows=_csv(directory/"episodes.csv",hashes)
                if len(rows)!=config["experiment"]["test_episodes"]:
                    raise ValueError("Test episode count differs from the shared protocol")
                scenarios=[_scenario_key(row) for row in rows]
                if reference_scenarios is None:
                    reference_scenarios=scenarios
                if scenarios!=reference_scenarios:
                    raise ValueError("A/B test scenarios do not match exactly")
                result,angle_rows,reward_rows=summarize_evaluation(rows,directory,environment,group,seed,model,hashes)
                evaluations.append(result)
                signs.extend(angle_rows)
                shaping.extend(reward_rows)
    candidate.mkdir(parents=True,exist_ok=False)
    analysis_source = Path(__file__).resolve()
    (candidate/"analysis_source.py").write_bytes(analysis_source.read_bytes())
    tables={"training_summary":training,"training_windows":windows,"validation_history":validations,
        "ppo_updates":updates,"ppo_summary":ppo,"test_summary":evaluations,"angle_sign_capture":signs,"shaping_by_category":shaping}
    for filename,rows in tables.items():
        _write_csv(candidate/(filename+".csv"),rows)
    _plots(candidate,validations,windows,updates,evaluations,seeds)
    report=_report(candidate,training,evaluations,signs,shaping,seeds,budget,metadata)
    for path,digest in hashes.items():
        if _sha(path)!=digest:
            raise RuntimeError("Source changed during comparison "+path)
    manifest=dict(status="completed",output_root=str(output_root),baseline_root=str(baseline_root),
        seeds=seeds,budget_per_seed=budget,metadata=metadata,source_sha256=hashes,
        all_source_files_unchanged=True,matched_test_scenarios=len(reference_scenarios),
        distance_bin_definition="<5, [5,10], (10,30], (30,100], >100",
        common_reward="Original absolute-distance, survival, capture and failure reward, excluding near shaping",
        reported_training_capture_denominator="Completed episodes only",report=str(report))
    manifest["analysis_implementation_sha256"] = _sha(analysis_source)
    manifest["analysis_source"] = "analysis_source.py"
    (candidate/"comparison_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report
