"""Provenance, safe paths and non-destructive experiment navigation checks."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pytest
import yaml

from scripts.experiment_index import (
    GENERATED, load_registry, render_index, suggest_next_run, validate_registry, write_indexes,
)


@pytest.fixture
def registry(tmp_path):
    output = tmp_path / "outputs/sample"
    output.mkdir(parents=True)
    (output / "metadata.json").write_text("{}", encoding="utf-8")
    (output / "model.pt").write_bytes(b"preserved checkpoint bytes")
    (output / "evaluation.json").write_text("{}", encoding="utf-8")
    (output / "pytest.log").write_text("231 passed", encoding="utf-8")
    return {
        "schema_version": 1, "current_version": "v05",
        "versions": [{"id": "v04", "title": "旧奖励", "status": "历史", "description": "旧实验"},
                     {"id": "v05", "title": "当前奖励", "status": "当前", "description": "同方案的训练与评估"}],
        "runs": [{"id": "v05-r01", "version": "v05", "title": "实际训练", "path": "outputs/sample",
                  "metadata_path": "outputs/sample/metadata.json", "started_at_utc": "2026-10-25T06:00:00+00:00",
                  "seed": 0, "total_env_steps": 200000, "kind": "training", "report": None,
                  "notes": "100k归档属于同run"}],
        "events": [{"id": "e001", "version": "v05", "title": "只有报告日期", "path": "outputs/sample",
                    "kind": "analysis", "occurred_at_utc": None, "occurred_on": "2026-10-25",
                    "time_source": "报告日期，无精确时刻", "related_run": "v05-r01", "notes": ""}],
        "models": [{"id": "v05-r01-closest", "run": "v05-r01", "criterion": "最近距离最小",
                    "checkpoint": "outputs/sample/model.pt", "training_steps": 158368,
                    "sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
                    "evaluation_path": "outputs/sample/evaluation.json", "video": None, "notes": "未捕获"}],
        "test_history": [{"version": "v05", "run": "v05-r01", "passed": 231, "seconds": 13.73,
                          "source": "outputs/sample/pytest.log", "recorded_on": "2026-11-07", "notes": "记录"}],
        "covered_output_roots": ["outputs/sample"],
        "planned_versions": [{"id": "v06", "title": "待定", "status": "未训练", "notes": "不能列为已有训练"}],
    }


def _hash_tree(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def test_registry_rendering_preserves_evidence_and_resolves_links(registry, tmp_path):
    source = tmp_path / "registry.yaml"
    source.write_text(yaml.safe_dump(registry, allow_unicode=True), encoding="utf-8")
    before = _hash_tree(tmp_path)
    loaded = load_registry(source)
    assert validate_registry(loaded, tmp_path) == []
    index = tmp_path / "experiments/EXPERIMENT_INDEX.md"
    rendered = render_index(loaded, tmp_path, index)
    assert "2026-10-25 14:00:00" in rendered
    assert "只有日期或时间未知" in rendered and "无精确时刻" in rendered
    assert "测试演进" in rendered and "231 / 13.73" in rendered
    assert "未捕获" in rendered and "不代表泛化" in rendered
    for target in re.findall(r"\]\(<([^>]+)>\)", rendered):
        assert (index.parent / target).resolve().exists()
    assert _hash_tree(tmp_path) == before


def test_registry_rejects_duplicate_ids_and_cross_version_references(registry, tmp_path):
    registry["runs"].append(deepcopy(registry["runs"][0]))
    registry["events"][0]["version"] = "v04"
    registry["models"][0]["run"] = "v05-r99"
    errors = validate_registry(registry, tmp_path)
    assert any("duplicate runs id" in error for error in errors)
    assert any("version differs from related_run" in error for error in errors)
    assert any("unknown run" in error for error in errors)


@pytest.mark.parametrize("escape", ["../outside", "/outside", "C:/outside"])
def test_registry_rejects_escape_paths(registry, tmp_path, escape):
    registry["models"][0]["checkpoint"] = escape
    assert any("repository" in error for error in validate_registry(registry, tmp_path))


def test_hash_changes_and_unregistered_output_directories_are_detected(registry, tmp_path):
    (tmp_path / "outputs/sample/model.pt").write_bytes(b"changed bytes")
    (tmp_path / "outputs/unregistered").mkdir()
    errors = validate_registry(registry, tmp_path)
    assert any("SHA-256 mismatch" in error for error in errors)
    assert any("unregistered output directory" in error for error in errors)


@pytest.mark.parametrize("kind", ["warm_start_curriculum_training", "policy_retention_training"])
def test_pretrained_model_may_have_zero_new_curriculum_steps(registry, tmp_path, kind):
    registry["runs"][0].update(kind=kind, source_pretrained_env_steps=92160,
        warm_start_checkpoint="outputs/sample/model.pt",
        warm_start_checkpoint_sha256=registry["models"][0]["sha256"])
    registry["models"][0].update(training_steps=0, source_pretrained_env_steps=92160)
    assert validate_registry(registry, tmp_path) == []
    registry["models"][0]["source_pretrained_env_steps"] = 123
    assert any("invalid training_steps" in message for message in validate_registry(registry, tmp_path))


def test_zero_new_steps_without_verified_warm_origin_are_rejected(registry, tmp_path):
    registry["models"][0]["training_steps"] = 0
    assert any("invalid training_steps" in message for message in validate_registry(registry, tmp_path))


def test_next_run_is_dry(registry, tmp_path):
    (tmp_path / "outputs/v05/v05-r02_20260101_seed007").mkdir(parents=True)
    before = _hash_tree(tmp_path)
    directories = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_dir()}
    proposed = suggest_next_run(registry, "v05", seed=7, date_string="20261108", root=tmp_path)
    assert proposed["id"] == "v05-r03"
    assert proposed["path"] == "outputs/v05/v05-r03_20261108_seed007"
    assert "--seed 7" in proposed["command"] and proposed["path"] in proposed["command"]
    assert not (tmp_path / proposed["path"]).exists()
    assert _hash_tree(tmp_path) == before
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_dir()} == directories
    assert suggest_next_run(registry, "v04", date_string="20261108", root=tmp_path)["command"] is None
    with pytest.raises(ValueError, match="Unknown implemented version"):
        suggest_next_run(registry, "v06", root=tmp_path)
    with pytest.raises(ValueError):
        suggest_next_run(registry, "v05", date_string="20260230", root=tmp_path)


def test_environment_fingerprint_rejects_wrong_version(registry, tmp_path):
    snapshot = tmp_path / "outputs/sample/environment.yaml"
    environment = {"mission": {"attack_radius": 10}, "reward": {"capture_reward": 100}}
    snapshot.write_text(yaml.safe_dump(environment), encoding="utf-8")
    digest = hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    registry["runs"][0].update(environment_snapshot="outputs/sample/environment.yaml", environment_fingerprint=digest)
    registry["versions"][1]["environment_fingerprint"] = digest
    assert validate_registry(registry, tmp_path) == []
    registry["versions"][1]["environment_fingerprint"] = "0" * 64
    assert any("environment differs from its version" in error for error in validate_registry(registry, tmp_path))


def test_current_version_uses_matching_normalized_config(registry, tmp_path):
    config = {"network": {"actor_hidden_sizes": [128, 128]}, "ppo": {"gamma": 0.99},
              "normalization": {"observation": True, "global_state": True, "reward_scale": 1.0}}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    config_path = tmp_path / "configs/mappo_normalized.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    registry["current_version"] = "v06"
    registry["versions"].append({"id": "v06", "title": "输入归一化", "status": "当前",
                                 "training_config_path": "configs/mappo_normalized.yaml",
                                 "learning_fingerprint": digest})
    assert validate_registry(registry, tmp_path) == []
    before = _hash_tree(tmp_path)
    proposed = suggest_next_run(registry, "v06", date_string="20261108", root=tmp_path)
    assert "--config configs/mappo_normalized.yaml" in proposed["command"]
    assert "mappo_baseline.yaml" not in proposed["command"]
    assert proposed["id"] == "v06-r01"
    assert suggest_next_run(registry, "v05", root=tmp_path)["command"] is None
    assert _hash_tree(tmp_path) == before
    config["normalization"]["observation"] = False
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert any("learning_fingerprint differs" in error for error in validate_registry(registry, tmp_path))
    with pytest.raises(ValueError, match="learning_fingerprint differs"):
        suggest_next_run(registry, "v06", root=tmp_path)


def test_training_snapshot_learning_settings_must_match_version(registry, tmp_path):
    settings = {"network": {"actor_hidden_sizes": [128, 128]}, "ppo": {"gamma": 0.99},
                "normalization": {"observation": False, "global_state": False, "reward_scale": 1.0}}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    snapshot = tmp_path / "outputs/sample/training_config.yaml"
    training = dict(settings, seed=7, training={"total_env_steps": 20000}, output_dir="outputs/sample")
    snapshot.write_text(yaml.safe_dump(training), encoding="utf-8")
    registry["runs"][0].update(training_snapshot="outputs/sample/training_config.yaml",
                               learning_fingerprint=digest)
    registry["versions"][1]["learning_fingerprint"] = digest
    assert validate_registry(registry, tmp_path) == []
    training["ppo"]["gamma"] = 0.95
    snapshot.write_text(yaml.safe_dump(training), encoding="utf-8")
    errors = validate_registry(registry, tmp_path)
    assert any("learning_fingerprint mismatch" in error for error in errors)
    assert any("learning settings differ from its version" in error for error in errors)
    registry["runs"][0].pop("learning_fingerprint")
    assert any("learning settings differ from its version" in error for error in validate_registry(registry, tmp_path))
    registry["runs"][0].pop("training_snapshot")
    registry["runs"][0]["learning_fingerprint"] = "0" * 64
    assert any("learning settings differ from its version" in error for error in validate_registry(registry, tmp_path))


def test_write_index_only_changes_generated_navigation(registry, tmp_path):
    before = _hash_tree(tmp_path)
    targets = write_indexes(registry, tmp_path)
    assert {path.relative_to(tmp_path).as_posix() for path in targets} == {
        "experiments/EXPERIMENT_INDEX.md", "outputs/README.md"}
    for name, digest in before.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert all(path.read_text(encoding="utf-8").startswith(GENERATED) for path in targets)
    original = targets[0].read_bytes()
    write_indexes(registry, tmp_path)
    assert targets[0].read_bytes() == original
    targets[1].write_text("manual notes", encoding="utf-8")
    with pytest.raises(ValueError, match="non-generated"):
        write_indexes(registry, tmp_path)
    assert targets[0].read_bytes() == original
    assert targets[1].read_text(encoding="utf-8") == "manual notes"


def test_descriptive_output_root_preserves_version_and_run_ids(registry, tmp_path):
    output_root = "outputs/v05_ActiveProgress奖励"
    (tmp_path / output_root).mkdir()
    registry["covered_output_roots"].append(output_root)
    registry["versions"][1]["output_root"] = output_root
    assert validate_registry(registry, tmp_path) == []
    suggested = suggest_next_run(registry, "v05", date_string="20261108", root=tmp_path)
    assert suggested["id"] == "v05-r02"
    assert suggested["path"] == output_root + "/v05-r02_20261108_seed000"
    assert suggested["path"] in suggested["command"]
    assert not (tmp_path / suggested["path"]).exists()
    registry["versions"][1]["output_root"] = "outputs/sample/nested"
    with pytest.raises(ValueError, match="direct outputs child"):
        suggest_next_run(registry, "v05", root=tmp_path)


def _add_formal_version(registry, tmp_path):
    container = "outputs/formal_ppo_v1"
    directory = container + "/v17_active_near"
    (tmp_path / directory).mkdir(parents=True)
    registry["covered_output_roots"].append(container)
    normalization = {"scales": [20.0] * 16, "acceleration_kind": "applied"}
    environment = {"normalization": normalization}
    config = {"network": {"actor_hidden_sizes": [128, 128]}, "ppo": {"gamma": .99},
              "env_config": "interception.yaml", "training": {"num_envs": 8}}
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "interception.yaml").write_text(yaml.safe_dump(environment), encoding="utf-8")
    (configs / "formal.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_formal_ppo.py").write_text("# test entrypoint", encoding="utf-8")
    canonical = dict(network=config["network"], ppo=config["ppo"], normalization=normalization)
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    version = dict(id="v17", title="formal PPO", status="completed", description="formal setup",
                   output_root=directory, output_container=container, training_system="formal_ppo",
                   training_entrypoint="scripts.train_formal_ppo", training_config_path="configs/formal.yaml",
                   learning_fingerprint=fingerprint)
    registry["versions"].append(version)
    registry["current_version"] = "v17"
    return version, config, environment


def test_formal_normalization_is_read_from_environment_and_checked(registry, tmp_path):
    version, config, environment = _add_formal_version(registry, tmp_path)
    assert validate_registry(registry, tmp_path) == []
    proposed = suggest_next_run(registry, "v17", seed=1001, date_string="20261120", root=tmp_path)
    assert proposed["path"].startswith(version["output_root"] + "/v17-r01_")
    assert "scripts.train_formal_ppo" in proposed["command"]
    environment["normalization"]["scales"][0] = 999.0
    (tmp_path / "configs/interception.yaml").write_text(yaml.safe_dump(environment), encoding="utf-8")
    assert any("learning_fingerprint differs" in error for error in validate_registry(registry, tmp_path))


def test_formal_embedded_snapshot_checks_actual_normalization(registry, tmp_path):
    version, config, environment = _add_formal_version(registry, tmp_path)
    snapshot = tmp_path / "outputs/sample/formal_snapshot.yaml"
    config["environment_config"] = environment
    snapshot.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = registry["runs"][0]
    run.update(id="v17-r01", version="v17", training_snapshot="outputs/sample/formal_snapshot.yaml",
               learning_fingerprint=version["learning_fingerprint"])
    registry["events"][0].update(version="v17", related_run="v17-r01")
    registry["models"][0]["run"] = "v17-r01"
    registry["test_history"][0].update(version="v17", run="v17-r01")
    assert validate_registry(registry, tmp_path) == []
    config["environment_config"]["normalization"]["acceleration_kind"] = "realized"
    snapshot.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert any("learning settings differ" in error for error in validate_registry(registry, tmp_path))


def test_explicit_formal_container_cannot_escape_or_add_hidden_depth(registry, tmp_path):
    version, _, _ = _add_formal_version(registry, tmp_path)
    version["output_root"] += "/unexpected/deeper"
    (tmp_path / version["output_root"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="registered outputs container"):
        suggest_next_run(registry, "v17", root=tmp_path)
    version["output_root"] = "../outside"
    with pytest.raises(ValueError, match="escapes repository"):
        suggest_next_run(registry, "v17", root=tmp_path)


def test_multiple_formal_variants_are_not_silently_chosen(registry, tmp_path):
    version, _, _ = _add_formal_version(registry, tmp_path)
    version["training_variants"] = [{"name": "beta100", "training_config_path": "configs/formal.yaml"},
                                    {"name": "beta200", "training_config_path": "configs/formal.yaml"}]
    assert validate_registry(registry, tmp_path) == []
    before = _hash_tree(tmp_path)
    proposed = suggest_next_run(registry, "v17", root=tmp_path)
    assert proposed["command"] is None and "配比组" in proposed["note"]
    assert _hash_tree(tmp_path) == before
    version["training_variants"][1]["training_config_path"] = "configs/missing.yaml"
    assert any("invalid variant" in error for error in validate_registry(registry, tmp_path))


def test_auxiliary_outputs_are_explicitly_covered_without_training_runs(registry, tmp_path):
    (tmp_path / "outputs/checks").mkdir()
    before_runs = deepcopy(registry["runs"])
    registry["covered_output_roots"].append("outputs/checks")
    registry["auxiliary_outputs"] = [{"path": "outputs/checks", "notes": "工程测试工件"}]
    assert validate_registry(registry, tmp_path) == []
    rendered = render_index(registry, tmp_path)
    assert "工程与分析执行资料" in rendered and "工程测试工件" in rendered
    assert registry["runs"] == before_runs
    registry["auxiliary_outputs"][0]["path"] = "outputs/missing"
    assert any("auxiliary output" in message for message in validate_registry(registry, tmp_path))


def test_unmeasured_final_and_measured_zero_capture_are_distinct(registry, tmp_path):
    model = registry["models"][0]
    model["model_family"] = "formal_single_ppo"
    model["evaluation_summary"] = {}
    assert validate_registry(registry, tmp_path) == []
    assert "未做500场景测试；固定0°视频仅为单例" in render_index(registry, tmp_path)
    model.update(evaluation_summary={"episodes": 500, "captures": 0, "capture_rate": 0.0},
                 evaluation_angle_range_degrees=[-30, 30])
    rendered = render_index(registry, tmp_path)
    assert "-30°至30°；0/500捕获＝0.0%" in rendered
    assert "未做500场景测试；固定0°视频仅为单例" not in rendered
    assert validate_registry(registry, tmp_path) == []
    model["evaluation_summary"]["capture_rate"] = .5
    assert any("counts/rate are inconsistent" in m for m in validate_registry(registry, tmp_path))
