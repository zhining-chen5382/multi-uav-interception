"""Migration compatibility without changing frozen artifacts or research values."""

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import yaml

from utils.artifact_paths import canonicalize_storage_paths, resolve_archived_file, resolve_artifact_path


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "catalog.yaml"
    value = dict(schema_version=1, root="outputs/formal_ppo_v1", moves=[
        dict(old="old_run", new="v16/stage")])
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return tmp_path, path


def test_migrated_absolute_and_relative_paths_read_same_bytes(catalog):
    root, path = catalog
    target = root / "outputs/formal_ppo_v1/v16/stage/checkpoints/best.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"frozen-checkpoint")
    old = Path("outputs/formal_ppo_v1/old_run/checkpoints/best.pt")
    for source in (old, root / old):
        actual = resolve_artifact_path(source, root, path)
        assert actual == target
        assert actual.read_bytes() == b"frozen-checkpoint"
    assert not (root / old).exists()


def test_existing_original_has_priority_over_mapping(catalog):
    root, path = catalog
    original = root / "outputs/formal_ppo_v1/old_run/a.csv"
    original.parent.mkdir(parents=True)
    original.write_text("original", encoding="utf-8")
    assert resolve_artifact_path(original, root, path) == original


def test_prefix_requires_a_directory_boundary(catalog):
    root, path = catalog
    source = root / "outputs/formal_ppo_v1/old_run_extra/file.csv"
    assert resolve_artifact_path(source, root, path) == source


def test_missing_target_does_not_create_data(catalog):
    root, path = catalog
    target = resolve_artifact_path("outputs/formal_ppo_v1/old_run/missing.csv", root, path)
    assert target == root / "outputs/formal_ppo_v1/v16/stage/missing.csv"
    assert not target.exists()


def test_missing_catalog_leaves_unrelated_path_unchanged(tmp_path):
    path = tmp_path / "missing.csv"
    assert resolve_artifact_path(path, tmp_path) == path


@pytest.mark.parametrize("section,value", [
    ("root", "../outside"), ("root", "C:/outside"),
    ("old", "../outside"), ("new", "../outside"),
    ("new", "/outside"), ("new", "C:/outside"), ("old", "."),
])
def test_mapping_rejects_absolute_or_escaping_targets(tmp_path, section, value):
    config = dict(schema_version=1, root="outputs", moves=[dict(old="source", new="target")])
    if section == "root":
        config[section] = value
    else:
        config["moves"][0][section] = value
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_artifact_path("outputs/source/a.csv", tmp_path, path)


def test_resume_comparison_only_canonicalizes_known_storage_fields(catalog):
    root, path = catalog
    old = "outputs/formal_ppo_v1/old_run"
    new = "outputs/formal_ppo_v1/v16/stage"
    before = dict(experiment=dict(output_root=old, seeds=[1001]),
                  comparison=dict(baseline_root=old),
                  reward=dict(capture=500), network=dict(shape=[128, 128]),
                  notes=old, training=dict(total_env_steps=300000))
    snapshot = deepcopy(before)
    current = deepcopy(before)
    current["experiment"]["output_root"] = new
    current["comparison"]["baseline_root"] = new
    assert canonicalize_storage_paths(before, root, path) == canonicalize_storage_paths(current, root, path)
    assert before == snapshot
    current["reward"]["capture"] = 501
    assert canonicalize_storage_paths(before, root, path) != canonicalize_storage_paths(current, root, path)
    assert canonicalize_storage_paths(before, root, path)["notes"] == old


def test_copy_prior_checkpoints_resolves_frozen_output_dir(tmp_path, monkeypatch):
    from utils import formal_ppo_training as training

    source = tmp_path / "new_stage"
    output = tmp_path / "continuation"
    for directory in (source / "checkpoints", source / "validation/steps_000010",
                      output / "checkpoints", output / "validation"):
        directory.mkdir(parents=True)
    (source / "checkpoints/initial.pt").write_bytes(b"initial")
    (source / "checkpoints/checkpoint_000008.pt").write_bytes(b"before")
    (source / "checkpoints/checkpoint_000024.pt").write_bytes(b"after")
    (source / "validation/steps_000010/summary.json").write_text("{}", encoding="utf-8")
    frozen = dict(output_dir=str(tmp_path / "old_stage"), training_state=dict(env_steps=16))
    snapshot = deepcopy(frozen)
    monkeypatch.setattr(training, "resolve_artifact_path", lambda value: source)
    training._copy_prior_checkpoints(frozen, output)
    assert (output / "checkpoints/initial.pt").read_bytes() == b"initial"
    assert (output / "checkpoints/checkpoint_000008.pt").read_bytes() == b"before"
    assert not (output / "checkpoints/checkpoint_000024.pt").exists()
    assert (output / "validation/steps_000010/summary.json").read_text(encoding="utf-8") == "{}"
    assert frozen == snapshot


@pytest.mark.parametrize("backup_name", ["活跃入口原件", "文档原件"])
def test_frozen_hash_uses_exact_pre_migration_backup(catalog, backup_name):
    root, path = catalog
    live = root / "configs/example.yaml"
    live.parent.mkdir()
    live.write_bytes(b"path: new-location\n")
    backup = root / "outputs/formal_ppo_v1/_工程验收/目录整理_20261120" / backup_name / "configs/example.yaml"
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b"path: original-location\n")
    expected = hashlib.sha256(backup.read_bytes()).hexdigest()
    assert resolve_archived_file(live, expected, root, path) == backup
    assert live.read_bytes() == b"path: new-location\n"


def test_frozen_hash_uses_matching_source_capture(catalog):
    root, path = catalog
    snapshot = root / "outputs/formal_ppo_v1/v16/stage/source_snapshot/utils/example.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"original-code")
    expected = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert resolve_archived_file("utils/example.py", expected, root, path) == snapshot


def test_frozen_hash_rejects_every_nonmatching_copy(catalog):
    root, path = catalog
    live = root / "configs/example.yaml"
    live.parent.mkdir()
    live.write_bytes(b"changed")
    backup = root / "outputs/formal_ppo_v1/_工程验收/目录整理_20261120/活跃入口原件/configs/example.yaml"
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b"also changed")
    with pytest.raises(ValueError, match="frozen SHA-256"):
        resolve_archived_file(live, hashlib.sha256(b"original").hexdigest(), root, path)


def test_frozen_hash_accepts_migrated_identical_artifact(catalog):
    root, path = catalog
    target = root / "outputs/formal_ppo_v1/v16/stage/episode.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"recorded-trajectory")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert resolve_archived_file("outputs/formal_ppo_v1/old_run/episode.npz", digest, root, path) == target
