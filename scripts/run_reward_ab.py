"""Sequential three-seed, matched-300k reward A/B experiment."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import torch
import yaml

from algorithms.ppo import PPOAgent
from scripts.run_formal_experiment import _snapshot_sources, run_acceptance_smoke
from utils.formal_ppo_config import load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import create_scenarios, evaluate_angles, write_json
from utils.formal_ppo_reporting import postprocess_run
from utils.formal_ppo_training import train


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "outputs/formal_ppo_v1/v14_单机PPO_原始奖励/01_从零训练至300k"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(config, baseline_root):
    validate_readiness(config, expected_seed_count=3)
    baseline_root = Path(baseline_root).resolve()
    environment = deepcopy(config["environment_config"])
    shaping = environment["reward"].pop("best_so_far_shaping")
    protected = {}
    for seed in config["experiment"]["seeds"]:
        run = baseline_root / ("seed_%d" % seed)
        checkpoint = torch.load(run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
        if checkpoint["training_state"]["env_steps"] != config["training"]["total_env_steps"]:
            raise ValueError("Historical A budget differs from the B budget")
        if environment != checkpoint["environment_config"]:
            raise ValueError("B changes environment settings beyond the requested reward term")
        for key in ("network", "ppo", "training", "device", "torch_num_threads", "acceptance"):
            if config[key] != checkpoint["config"][key]:
                raise ValueError("B changes the original protocol " + key)
        for key in ("validation_seed", "test_seed", "test_episodes", "reference_angle_degrees", "angle_bins_degrees"):
            if config["experiment"][key] != checkpoint["config"]["experiment"][key]:
                raise ValueError("B changes held-out evaluation " + key)
        for name in ("summary.json", "environment_config.yaml", "training_config.yaml", "episodes.csv",
                     "updates.csv", "validation.csv", "checkpoints/initial.pt", "checkpoints/best.pt",
                     "checkpoints/final.pt", "final_test/episodes.csv"):
            path = run / name
            protected[str(path)] = _sha(path)
    return dict(ready=True, baseline_root=str(baseline_root), seeds=config["experiment"]["seeds"],
                budget_per_seed=config["training"]["total_env_steps"], shaping=shaping,
                only_reward_changed=True, historical_source_sha256=protected)


def _prepare(config, output, baseline_root, checked):
    if output.exists():
        if yaml.safe_load((output / "experiment_config.yaml").read_text(encoding="utf-8")) != config:
            raise ValueError("Existing experiment configuration differs")
        return _json(output / "scenarios/scenario_manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    scenarios = create_scenarios(config, config["environment_config"], output / "scenarios")
    historical = _json(baseline_root / "scenarios/scenario_manifest.json")
    for key in ("validation_sha256", "test_sha256"):
        if scenarios[key] != historical[key]:
            raise ValueError("A/B scenario sets differ")
    _snapshot_sources(output, config)
    manifest = _json(output / "source_manifest.json")
    for relative in ("scripts/run_reward_ab.py", "scripts/verify_reward_ab_acceptance.py",
                     "utils/reward_ab_analysis.py", "configs/ppo_reward_ab.yaml",
                     "configs/interception_1v1_best_so_far.yaml", "tests/test_best_so_far_training_reward.py",
                     "tests/test_reward_ab_training.py", "tests/test_reward_ab_runner.py", "tests/test_reward_ab_analysis.py"):
        source, destination = ROOT / relative, output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest["files"].append(dict(path=relative, sha256=_sha(source)))
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(output / "CONTROLLED_PREFLIGHT.json", checked)
    return scenarios


def finish_seed(config, output, seed, train_fn=train, postprocess_fn=postprocess_run):
    run = output / ("seed_%d" % seed)
    if (run / "summary.json").is_file():
        summary = _json(run / "summary.json")
    else:
        if run.exists():
            raise RuntimeError("Partial run requires exact checkpoint recovery into a fresh directory " + str(run))
        summary = train_fn(config, run, seed)
    if summary["status"] != "completed" or summary["env_steps"] != config["training"]["total_env_steps"]:
        raise RuntimeError("The seed did not complete its configured budget")
    for path in sorted(run.glob("postprocessing_*/postprocessing_manifest.json"), reverse=True):
        if _json(path).get("status") == "completed":
            return summary, _json(path)
    return summary, postprocess_fn(run, config, summary)


def evaluate_checkpoint(checkpoint_path, output, angles, seed, evaluation_fn=evaluate_angles):
    if (output / "summary.json").is_file():
        return _json(output / "summary.json")
    if output.exists():
        raise RuntimeError("Preserve and inspect the unfinished evaluation " + str(output))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = PPOAgent(checkpoint["config"])
    agent.load_state_dict(checkpoint["agent"])
    metrics = evaluation_fn(agent, checkpoint["environment_config"], angles, seed, output)
    write_json(output / "MODEL_SOURCE.json", dict(checkpoint=str(checkpoint_path.resolve()),
        sha256=_sha(checkpoint_path), checkpoint_steps=checkpoint["training_state"]["env_steps"], deterministic=True))
    return metrics


def run_experiment(config, output, baseline_root=BASELINE_ROOT, train_fn=train,
                   postprocess_fn=postprocess_run, evaluation_fn=evaluate_angles, analysis_fn=None):
    config = deepcopy(config)
    output, baseline_root = Path(output).resolve(), Path(baseline_root).resolve()
    checked = preflight(config, baseline_root)
    scenarios = _prepare(config, output, baseline_root, checked)
    summaries, postprocessing = [], []
    for seed in config["experiment"]["seeds"]:
        summary, post = finish_seed(config, output, seed, train_fn, postprocess_fn)
        summaries.append(summary)
        postprocessing.append(post)
    for seed in config["experiment"]["seeds"]:
        run = output / ("seed_%d" % seed)
        for role, directory in (("best", "final_test"), ("final", "final_model_test")):
            evaluate_checkpoint(run / ("checkpoints/%s.pt" % role), run / directory,
                scenarios["test_angles"], config["experiment"]["test_seed"], evaluation_fn)
        historical_final = baseline_root / ("seed_%d/final_model_test/summary.json" % seed)
        if not historical_final.is_file():
            evaluate_checkpoint(baseline_root / ("seed_%d/checkpoints/final.pt" % seed),
                output / ("baseline_evaluation/seed_%d/final" % seed), scenarios["test_angles"],
                config["experiment"]["test_seed"], evaluation_fn)
    if analysis_fn is None:
        from utils.reward_ab_analysis import analyze_experiment
        analysis_fn = analyze_experiment
    report = analysis_fn(output, baseline_root)
    for path, digest in checked["historical_source_sha256"].items():
        if _sha(path) != digest:
            raise RuntimeError("Historical A artifact changed " + path)
    completion = dict(status="completed", seeds=config["experiment"]["seeds"],
        training_steps=sum(s["env_steps"] for s in summaries), summaries=summaries,
        postprocessing=postprocessing, comparison_report=str(report), historical_a_unchanged=True)
    if not (output / "COMPLETED.json").exists():
        write_json(output / "COMPLETED.json", completion)
    lines = ["# Reward A/B，三个Seed各300k", "", "[完整对比报告](%s)" % Path(report).relative_to(output).as_posix(), "",
             "A组为同三个Seed各300k历史结果，B组使用λ20m、β100历史最小距离shaping。", "",
             "| Seed | 本轮报告 | XY演化 | Best视频 | Final视频 |", "| --- | --- | --- | --- | --- |"]
    for seed, post in zip(config["experiment"]["seeds"], postprocessing):
        relative = Path(post["output_dir"]).relative_to(output).as_posix()
        lines.append("| %d | [报告](%s/TRAINING_REPORT.md) | [XY](%s/trajectory_evolution/trajectory_evolution_xy.png) | [Best](%s/videos/best/best_full_episode.mp4) | [Final](%s/videos/final/final_full_episode.mp4) |" % (seed, relative, relative, relative, relative))
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo_reward_ab.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_formal_config(args.config)
    checked = preflight(config, args.baseline_root)
    if args.check_config:
        print(json.dumps({key:value for key,value in checked.items() if key != "historical_source_sha256"}, indent=2))
        return
    output = Path(args.output_dir or config["experiment"]["output_root"]).resolve()
    smoke = args.smoke_output or output.parent / (output.name + "_acceptance_smoke")
    if (smoke / "SMOKE_RESULT.json").is_file():
        saved = yaml.safe_load((smoke / "uninterrupted/training_config.yaml").read_text(encoding="utf-8"))
        for key in ("network", "ppo", "training", "environment_config"):
            if saved[key] != config[key]:
                raise ValueError("Saved acceptance smoke used different parameters " + key)
        if not _json(smoke / "SMOKE_RESULT.json")["passed"]:
            raise RuntimeError("Production smoke did not pass")
    else:
        run_acceptance_smoke(config, smoke, expected_seed_count=3)
    result = run_experiment(config, output, args.baseline_root)
    print(json.dumps(dict(status=result["status"], training_steps=result["training_steps"],
                         comparison_report=result["comparison_report"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
