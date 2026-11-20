"""Locate reorganized artifacts without rewriting historical provenance.

Only explicit directory-prefix mappings in the formal output catalog apply.
Existing paths retain priority; no data is moved or modified by this module.
"""

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path, PureWindowsPath
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path("experiments/formal_ppo_output_catalog.yaml")
CONFIG_STORAGE_FIELDS = (("experiment", "output_root"), ("comparison", "baseline_root"))


def _relative(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or Path(value).is_absolute() or PureWindowsPath(value).drive
            or PureWindowsPath(value).root or ".." in value.replace("\\", "/").split("/")):
        raise ValueError("Artifact mapping must stay repository-relative: {!r}".format(value))
    return Path(value)


@lru_cache(maxsize=8)
def _read_catalog(path, modification_time):
    # The mtime cache key permits another process to atomically refresh a catalog.
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported artifact catalog")
    base = _relative(data.get("root"))
    moves = data.get("moves")
    if not isinstance(moves, list):
        raise ValueError("Artifact catalog moves must be a list")
    pairs, sources = [], set()
    for item in moves:
        if not isinstance(item, dict):
            raise ValueError("Invalid artifact mapping")
        source, destination = base / _relative(item.get("old")), base / _relative(item.get("new"))
        if source in sources:
            raise ValueError("Duplicate artifact mapping source")
        sources.add(source)
        pairs.append((source, destination))
    return tuple(sorted(pairs, key=lambda pair: len(pair[0].parts), reverse=True))


def _pairs(root, catalog_path):
    path = root / CATALOG if catalog_path is None else Path(catalog_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return ()
    return _read_catalog(str(path.resolve()), path.stat().st_mtime_ns)


def _mapped(path, root, catalog_path):
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path
    for old, new in _pairs(root, catalog_path):
        # Path boundaries matter: source 'run' must not match 'run_extra'.
        try:
            suffix = relative.relative_to(old)
        except ValueError:
            continue
        target = (root / new / suffix).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ValueError("Artifact mapping resolves outside the repository")
        return target
    return path


def resolve_artifact_path(value, root=ROOT, catalog_path=None):
    """Resolve an existing path or its explicitly registered migrated location.

    Missing unrelated paths remain missing. Mapped paths are returned even if
    their target is missing, so the caller's normal existence check still fails.
    An existing original (including a test fixture) always takes precedence.
    """
    root = Path(root).resolve()
    candidate = Path(value)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    return candidate if candidate.exists() else _mapped(candidate, root, catalog_path)


def canonicalize_storage_paths(config, root=ROOT, catalog_path=None):
    """Copy a configuration and canonicalize only two storage-path fields.

    This is for strict resume comparisons: reward, environment, learning
    settings, seeds and budgets retain exact equality. The saved configuration
    and Checkpoint are never mutated, and absent optional fields stay absent.
    """
    root = Path(root).resolve()
    result = deepcopy(config)
    for section, field in CONFIG_STORAGE_FIELDS:
        mapping = result.get(section)
        if isinstance(mapping, dict) and isinstance(mapping.get(field), str):
            path = Path(mapping[field])
            path = (path if path.is_absolute() else root / path).resolve()
            mapping[field] = str(_mapped(path, root, catalog_path))
    return result


def resolve_archived_file(value, expected_sha256, root=ROOT, catalog_path=None):
    """Find the exact original bytes demanded by a frozen provenance hash.

    Path-only edits to active scripts/configs do not invalidate an archived
    experiment: migration backups and captured source trees may retain its
    original bytes. Every candidate must match the supplied hash; otherwise
    the original integrity failure remains an explicit exception.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("An exact lowercase SHA-256 is required")
    root = Path(root).resolve()
    original = Path(value)
    original = (original if original.is_absolute() else root / original).resolve()
    candidates = [resolve_artifact_path(original, root, catalog_path),
                  _mapped(original, root, catalog_path)]
    try:
        relative = original.relative_to(root)
    except ValueError:
        relative = None
    if relative is not None:
        archive = root / "outputs/formal_ppo_v1/_工程验收"
        if archive.exists():
            for backup_name in ("活跃入口原件", "文档原件"):
                candidates.extend(path / relative for path in sorted(archive.glob("目录整理_*/" + backup_name)))
        for old, new in _pairs(root, catalog_path):
            candidates.extend((root / prefix / "source_snapshot" / relative) for prefix in (new, old))
    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked or not candidate.is_file():
            continue
        checked.add(candidate)
        # Backups/source captures must stay in the intended workspace.
        if candidate not in candidates[:2]:
            try:
                candidate.relative_to(root)
            except ValueError:
                raise ValueError("Archived source resolves outside the repository")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256:
            return candidate
    raise ValueError("No artifact matches the frozen SHA-256: " + str(value))
