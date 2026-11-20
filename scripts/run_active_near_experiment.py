"""Run two matched, sequential three-seed motion-attributed-near experiments."""

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
from utils.formal_ppo_training import _state_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (ROOT / "configs/ppo_active_near_beta100.yaml", ROOT / "configs/ppo_active_near_beta200.yaml")
EXTRA_SOURCES = (
    "envs/geometry.py", "envs/near_shaping.py", "utils/active_near_replay.py",
    "scripts/run_active_near_experiment.py", "utils/active_near_experiment_analysis.py",
    "configs/ppo_active_near_beta100.yaml", "configs/ppo_active_near_beta200.yaml",
    "configs/interception_1v1_active_near_beta100.yaml", "configs/interception_1v1_active_near_beta200.yaml",
    "tests/test_active_near_experiment.py", "tests/test_motion_attributed_near_reward.py",
    "utils/formal_trajectory_overview.py", "configs/formal_trajectory_overview.yaml",
    "scripts/plot_formal_trajectory_overview.py",
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_hashes(hashes):
    for path, digest in hashes.items():
        if _sha(path) != digest:
            raise RuntimeError("Historical reference changed " + path)


def preflight(config, baseline_root):
    """Check every preserved protocol setting against the previous final models."""
    validate_readiness(config, expected_seed_count=3)
    baseline_root = Path(baseline_root).resolve()
    environment = deepcopy(config["environment_config"])
    shaping = environment["reward"]["best_so_far_shaping"]
    if shaping.pop("mode", None) != "motion_attributed":
        raise ValueError("The new experiment requires motion_attributed shaping")
    beta = shaping["beta"]
    coefficient = environment["reward"]["line_cost_coefficient"]
    protected = {}
    for seed in config["experiment"]["seeds"]:
        directory = baseline_root / ("seed_%d" % seed)
        checkpoint = torch.load(directory / "checkpoints/final.pt", map_location="cpu", weights_only=False)
        if checkpoint["training_state"]["env_steps"] != config["training"]["total_env_steps"]:
            raise ValueError("Historical reference budget differs")
        expected = deepcopy(environment)
        old_reward = checkpoint["environment_config"]["reward"]
        expected["reward"]["best_so_far_shaping"]["beta"] = old_reward["best_so_far_shaping"]["beta"]
        expected["reward"]["line_cost_coefficient"] = old_reward["line_cost_coefficient"]
        if expected != checkpoint["environment_config"]:
            raise ValueError("Environment changes extend beyond near attribution, beta and line coefficient")
        for field in ("network", "ppo", "training", "device", "torch_num_threads", "acceptance"):
            if config[field] != checkpoint["config"][field]:
                raise ValueError("Training protocol changed " + field)
        experiment = deepcopy(checkpoint["config"]["experiment"])
        experiment["output_root"] = config["experiment"]["output_root"]
        if config["experiment"] != experiment:
            raise ValueError("Training seeds or evaluation settings changed")
        for relative in ("summary.json", "environment_config.yaml", "training_config.yaml", "episodes.csv",
                         "updates.csv", "validation.csv", "checkpoints/initial.pt", "checkpoints/best.pt",
                         "checkpoints/final.pt", "final_test/episodes.csv", "final_model_test/episodes.csv"):
            path = directory / relative
            protected[str(path.resolve())] = _sha(path)
    for relative in ("scenario_manifest.json", "validation_scenarios.csv", "test_scenarios.csv"):
        path = baseline_root / "scenarios" / relative
        protected[str(path.resolve())] = _sha(path)
    return dict(ready=True, beta=beta, line_coefficient=coefficient,
                baseline_root=str(baseline_root), seeds=config["experiment"]["seeds"],
                steps_per_seed=config["training"]["total_env_steps"], only_reward_changed=True,
                historical_source_sha256=protected)


def prepare_group(config, output, baseline_root, checked):
    output = Path(output)
    existing = output.exists()
    if existing:
        if yaml.safe_load((output / "experiment_config.yaml").read_text(encoding="utf-8")) != config:
            raise ValueError("Existing group configuration differs")
        scenarios = _json(output / "scenarios/scenario_manifest.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "experiment_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        scenarios = create_scenarios(config, config["environment_config"], output / "scenarios")
    historical = _json(Path(baseline_root) / "scenarios/scenario_manifest.json")
    for field in ("validation_sha256", "test_sha256"):
        if scenarios[field] != historical[field]:
            raise ValueError("Historical and new fixed scene sets differ")
    if existing:
        return scenarios
    _snapshot_sources(output, config)
    manifest = _json(output / "source_manifest.json")
    for relative in EXTRA_SOURCES:
        source, destination = ROOT / relative, output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = _sha(source)
        if _sha(destination) != digest:
            raise RuntimeError("Source snapshot hash mismatch " + relative)
        manifest["files"].append(dict(path=relative, sha256=digest))
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(output / "CONTROLLED_PREFLIGHT.json", checked)
    return scenarios


def ensure_smoke(config, output):
    result_path = output / "SMOKE_RESULT.json"
    if not result_path.is_file():
        return run_acceptance_smoke(config, output, expected_seed_count=3)
    saved = yaml.safe_load((output / "uninterrupted/training_config.yaml").read_text(encoding="utf-8"))
    for field in ("network", "ppo", "training", "environment_config"):
        if saved[field] != config[field]:
            raise ValueError("Existing acceptance smoke used different settings " + field)
    result = _json(result_path)
    if not result["passed"]:
        raise RuntimeError("Acceptance smoke did not pass")
    return result


def run_group(config, output, baseline_root, checked):
    scenarios = prepare_group(config, output, baseline_root, checked)
    completion = output / "COMPLETED.json"
    expected_steps = len(config["experiment"]["seeds"]) * config["training"]["total_env_steps"]
    if completion.is_file():
        result = _json(completion)
        if result["status"] != "completed" or result["training_steps"] != expected_steps:
            raise ValueError("Group completion has a different training budget")
        _verify_hashes(checked["historical_source_sha256"])
        return result
    summaries, posts = [], []
    for seed in config["experiment"]["seeds"]:
        # finish_seed preserves partial runs and requires explicit exact recovery.
        summary, post = finish_seed(config, output, seed)
        summaries.append(summary)
        posts.append(post)
    for seed in config["experiment"]["seeds"]:
        directory = output / ("seed_%d" % seed)
        for role, folder in (("best", "final_test"), ("final", "final_model_test")):
            evaluate_checkpoint(directory / ("checkpoints/%s.pt" % role), directory / folder,
                                scenarios["test_angles"], config["experiment"]["test_seed"])
    _verify_hashes(checked["historical_source_sha256"])
    result = dict(status="completed", seeds=config["experiment"]["seeds"], training_steps=expected_steps,
                  summaries=summaries, postprocessing=posts, historical_reference_unchanged=True)
    write_json(completion, result)
    lines = ["# 校正near训练组", "", "β=%g，k=%g，λ=%gm。" % (
        checked["beta"], checked["line_coefficient"], config["environment_config"]["reward"]["best_so_far_shaping"]["length_scale_m"]), "",
        "| Seed | 训练报告 | XY演化 | Best完整视频 | Final完整视频 |", "| --- | --- | --- | --- | --- |"]
    for seed, post in zip(config["experiment"]["seeds"], posts):
        relative = Path(post["output_dir"]).relative_to(output).as_posix()
        lines.append("| %d | [报告](%s/TRAINING_REPORT.md) | [XY](%s/trajectory_evolution/trajectory_evolution_xy.png) | [Best](%s/videos/best/best_full_episode.mp4) | [Final](%s/videos/final/final_full_episode.mp4) |" % (
            seed, relative, relative, relative, relative))
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def verify_initialization(group_outputs, baseline_root, seeds):
    records = []
    for seed in seeds:
        paths = [Path(baseline_root) / ("seed_%d/checkpoints/initial.pt" % seed)]
        paths.extend(directory / ("seed_%d/checkpoints/initial.pt" % seed) for directory in group_outputs.values())
        agents = []
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            agent = checkpoint["agent"]
            if checkpoint["training_state"]["env_steps"] != 0 or agent["update_count"] != 0:
                raise RuntimeError("Initial checkpoint already contains training updates")
            for name in ("actor_optimizer", "critic_optimizer"):
                if agent[name]["state"]:
                    raise RuntimeError("Fresh initialization contains Adam moments")
            agents.append(agent)
        for name in ("actor", "critic"):
            if len({_state_hash(agent[name]) for agent in agents}) != 1:
                raise RuntimeError("Matched-seed initial networks differ: " + name)
        records.append(dict(seed=seed, actor_critic_bitwise_identical=True, empty_adam_states=True,
                            checkpoint_sha256={str(path): _sha(path) for path in paths}))
    return dict(passed=True, reference="Historical k=0.1 and both fresh groups", records=records)


def load_configs():
    return [load_formal_config(path) for path in DEFAULT_CONFIGS]


def run_experiment(configs=None, output_root=None, baseline_root=None, analysis_fn=None, smoke_only=False):
    configs = deepcopy(load_configs() if configs is None else configs)
    output_root = Path(output_root or Path(configs[0]["experiment"]["output_root"]).parent).resolve()
    baseline_root = Path(baseline_root or ROOT / configs[0]["comparison"]["baseline_root"]).resolve()
    group_names = [Path(config["experiment"]["output_root"]).name for config in configs]
    if len(configs) != 2 or len(set(group_names)) != 2:
        raise ValueError("Two independently named experiment groups are required")
    checks = [preflight(config, baseline_root) for config in configs]
    suite_config = {name: config for name, config in zip(group_names, configs)}
    suite_path = output_root / "suite_config.yaml"
    if output_root.exists():
        if not suite_path.is_file() or yaml.safe_load(suite_path.read_text(encoding="utf-8")) != suite_config:
            raise ValueError("Existing suite configuration differs")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        suite_path.write_text(yaml.safe_dump(suite_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    completion = output_root / "COMPLETED.json"
    if completion.is_file():
        result = _json(completion)
        expected = sum(len(c["experiment"]["seeds"]) * c["training"]["total_env_steps"] for c in configs)
        if result["status"] != "completed" or result["training_steps"] != expected:
            raise ValueError("Suite completion has a different training budget")
        for checked in checks:
            _verify_hashes(checked["historical_source_sha256"])
        return result
    # Both production recovery checks complete before either formal group starts.
    for name, config in zip(group_names, configs):
        ensure_smoke(config, output_root / "acceptance" / name)
    if smoke_only:
        return dict(status="acceptance_passed", training_steps=0, output_root=str(output_root))
    group_outputs, posts, results = {}, {}, {}
    for name, config, checked in zip(group_names, configs, checks):
        directory = output_root / name
        result = run_group(config, directory, baseline_root, checked)
        group_outputs[name], posts[name], results[name] = directory, result["postprocessing"], result
    initialization = verify_initialization(group_outputs, baseline_root, configs[0]["experiment"]["seeds"])
    initialization_path = output_root / "INITIALIZATION_CHECK.json"
    if initialization_path.is_file():
        if _json(initialization_path) != initialization:
            raise RuntimeError("Stored initialization check differs")
    else:
        write_json(initialization_path, initialization)
    if analysis_fn is None:
        from utils.active_near_experiment_analysis import analyze
        analysis_fn = analyze
    report = Path(analysis_fn(output_root, group_outputs, baseline_root, posts)).resolve()
    for checked in checks:
        _verify_hashes(checked["historical_source_sha256"])
    result = dict(status="completed", training_steps=sum(r["training_steps"] for r in results.values()),
                  groups=results, group_outputs={name: str(path) for name, path in group_outputs.items()},
                  comparison_report=str(report), initialization_check=initialization,
                  historical_reference_unchanged=True)
    write_json(completion, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    configs = load_configs()
    if args.check_config:
        checked = [preflight(config, args.baseline_root or ROOT / config["comparison"]["baseline_root"]) for config in configs]
        print(json.dumps([{k: v for k, v in row.items() if k != "historical_source_sha256"} for row in checked], indent=2))
        return
    result = run_experiment(configs, args.output_dir, args.baseline_root, smoke_only=args.smoke_only)
    print(json.dumps(dict(status=result["status"], training_steps=result["training_steps"],
                         report=result.get("comparison_report")), ensure_ascii=False))


if __name__ == "__main__":
    main()
