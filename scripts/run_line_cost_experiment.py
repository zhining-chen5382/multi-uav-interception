"""Run the independent, controlled three-seed interception-line experiment."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import torch
import yaml

from scripts.run_formal_experiment import _snapshot_sources, run_acceptance_smoke
from scripts.run_reward_ab import finish_seed, evaluate_checkpoint
from utils.formal_ppo_config import load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import create_scenarios, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/ppo_line_cost_k0p1.yaml"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(config, baseline_root):
    validate_readiness(config, expected_seed_count=3)
    environment = deepcopy(config["environment_config"])
    coefficient = environment["reward"].pop("line_cost_coefficient")
    protected = {}
    for seed in config["experiment"]["seeds"]:
        directory = Path(baseline_root) / ("seed_%d" % seed)
        checkpoint = torch.load(directory / "checkpoints/final.pt", map_location="cpu", weights_only=False)
        if environment != checkpoint["environment_config"]:
            raise ValueError("Experiment changes the previous B environment beyond the line cost")
        for name in ("network", "ppo", "training", "device", "torch_num_threads", "acceptance"):
            if config[name] != checkpoint["config"][name]:
                raise ValueError("Experiment changes the previous B protocol " + name)
        expected_experiment = deepcopy(checkpoint["config"]["experiment"])
        expected_experiment["output_root"] = config["experiment"]["output_root"]
        if config["experiment"] != expected_experiment:
            raise ValueError("Experiment changes seed or evaluation settings")
        for name in ("summary.json", "environment_config.yaml", "training_config.yaml", "episodes.csv",
                     "updates.csv", "validation.csv", "checkpoints/initial.pt", "checkpoints/best.pt",
                     "checkpoints/final.pt", "final_test/episodes.csv", "final_model_test/episodes.csv"):
            path = directory / name
            protected[str(path.resolve())] = _sha(path)
    return dict(ready=True, baseline_root=str(Path(baseline_root).resolve()),
                coefficient=coefficient, seeds=config["experiment"]["seeds"],
                steps_per_seed=config["training"]["total_env_steps"], only_reward_changed=True,
                historical_source_sha256=protected)


def prepare(config, output, baseline_root, checked):
    output = Path(output)
    if output.exists():
        if yaml.safe_load((output / "experiment_config.yaml").read_text(encoding="utf-8")) != config:
            raise ValueError("Existing output has a different resolved configuration")
        return _json(output / "scenarios/scenario_manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    scenarios = create_scenarios(config, config["environment_config"], output / "scenarios")
    historical = _json(Path(baseline_root) / "scenarios/scenario_manifest.json")
    for field in ("validation_sha256", "test_sha256"):
        if scenarios[field] != historical[field]:
            raise ValueError("Matched scenario sets changed")
    _snapshot_sources(output, config)
    source_manifest = _json(output / "source_manifest.json")
    for relative in ("envs/geometry.py", "scripts/run_line_cost_experiment.py",
                     "utils/line_cost_experiment_analysis.py", "configs/ppo_line_cost_k0p1.yaml",
                     "configs/interception_1v1_line_cost_k0p1.yaml", "tests/test_line_cost_training_reward.py",
                     "utils/formal_trajectory_overview.py", "configs/formal_trajectory_overview.yaml",
                     "scripts/plot_formal_trajectory_overview.py"):
        source, destination = ROOT / relative, output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_manifest["files"].append(dict(path=relative, sha256=_sha(source)))
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(output / "CONTROLLED_PREFLIGHT.json", checked)
    (output / "README.md").write_text(
        "# k=0.1线段成本独立实验\n\n"
        "三个Seed各从零300k，按1001、1002、1003顺序运行。\n\n"
        "原四项与历史最小距离shaping基础上增加三维闭线段成本。\n\n"
        "每个seed目录保存训练CSV、Checkpoint、验证、轨迹演化与Best/Final完整视频。\n",
        encoding="utf-8")
    return scenarios


def run_experiment(config, output, baseline_root):
    checked = preflight(config, baseline_root)
    scenarios = prepare(config, output, baseline_root, checked)
    completion_path = Path(output) / "COMPLETED.json"
    if completion_path.is_file():
        completed = _json(completion_path)
        expected_steps = len(config["experiment"]["seeds"]) * config["training"]["total_env_steps"]
        if completed["status"] != "completed" or completed["training_steps"] != expected_steps:
            raise ValueError("Stored completion does not match the experiment budget")
        for path, digest in checked["historical_source_sha256"].items():
            if _sha(path) != digest:
                raise RuntimeError("Historical reference changed " + path)
        return completed
    smoke = Path(output) / "acceptance/production_smoke"
    if not (smoke / "SMOKE_RESULT.json").is_file():
        run_acceptance_smoke(config, smoke, expected_seed_count=3)
    else:
        saved = yaml.safe_load((smoke / "uninterrupted/training_config.yaml").read_text(encoding="utf-8"))
        for key in ("network", "ppo", "training", "environment_config"):
            if saved[key] != config[key]:
                raise ValueError("Acceptance smoke used different settings " + key)
        if not _json(smoke / "SMOKE_RESULT.json")["passed"]:
            raise RuntimeError("Acceptance smoke did not pass")
    summaries, posts = [], []
    for seed in config["experiment"]["seeds"]:
        summary, post = finish_seed(config, Path(output), seed)
        summaries.append(summary)
        posts.append(post)
    for seed in config["experiment"]["seeds"]:
        directory = Path(output) / ("seed_%d" % seed)
        for role, folder in (("best", "final_test"), ("final", "final_model_test")):
            evaluate_checkpoint(directory / ("checkpoints/%s.pt" % role), directory / folder,
                                scenarios["test_angles"], config["experiment"]["test_seed"])
    from utils.line_cost_experiment_analysis import analyze
    report = analyze(Path(output), Path(baseline_root), posts)
    for path, digest in checked["historical_source_sha256"].items():
        if _sha(path) != digest:
            raise RuntimeError("Historical reference changed " + path)
    result = dict(status="completed", seeds=config["experiment"]["seeds"],
                  training_steps=sum(summary["env_steps"] for summary in summaries),
                  summaries=summaries, postprocessing=posts, comparison_report=str(report),
                  historical_reference_unchanged=True)
    if not (Path(output) / "COMPLETED.json").exists():
        write_json(Path(output) / "COMPLETED.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_formal_config(args.config)
    baseline = ROOT / config["comparison"]["baseline_root"]
    if args.check_config:
        checked = preflight(config, baseline)
        print(json.dumps({key: value for key, value in checked.items()
                          if key != "historical_source_sha256"}, ensure_ascii=False, indent=2))
        return
    result = run_experiment(config, ROOT / config["experiment"]["output_root"], baseline)
    print(json.dumps(dict(status=result["status"], training_steps=result["training_steps"],
                         report=result["comparison_report"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
