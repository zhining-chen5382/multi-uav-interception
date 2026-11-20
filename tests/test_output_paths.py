"""Versioned output path resolution preserves archived records."""

import ast
from pathlib import Path

import pytest

from utils.output_paths import resolve_recorded_path, translate_relative_path, translate_script_paths


def test_complete_prefix_mapping_preserves_suffix_and_similar_sibling():
    mappings = {"outputs/old": "outputs/v05_ActiveProgress奖励/run"}
    assert translate_relative_path("outputs/old/checkpoints/final.pt", mappings) == "outputs/v05_ActiveProgress奖励/run/checkpoints/final.pt"
    assert translate_relative_path("outputs\\old\\evaluation", mappings) == "outputs/v05_ActiveProgress奖励/run/evaluation"
    assert translate_relative_path("outputs/old_extra/final.pt", mappings) == "outputs/old_extra/final.pt"
    assert translate_relative_path("outputs/v05_ActiveProgress奖励/run/final.pt", mappings) == "outputs/v05_ActiveProgress奖励/run/final.pt"


def test_recorded_relative_and_absolute_paths_resolve_without_mutation(tmp_path):
    expected = tmp_path / "outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoints/final.pt"
    relative = "outputs/active_reward_100k/checkpoints/final.pt"
    assert resolve_recorded_path(relative, tmp_path) == expected
    assert resolve_recorded_path(tmp_path / relative, tmp_path) == expected
    assert not expected.exists()


def test_recorded_path_cannot_escape_repository(tmp_path):
    for value in ("../outside", "outputs/../outside", "C:/outside"):
        with pytest.raises(ValueError):
            translate_relative_path(value)
    with pytest.raises(ValueError):
        resolve_recorded_path(tmp_path.parent / "outside", tmp_path)


def test_script_adapter_changes_path_literals_only():
    source = "a = ROOT / 'outputs' / 'old' / 'metrics.csv'\nb = 'outputs/old/final.pt'\nvalue = 0.99 * reward + 100\n"
    original = str(source)
    tree = translate_script_paths(source, {"outputs/old": "outputs/v05_ActiveProgress奖励/run"})
    namespace = {"ROOT": Path("repo"), "reward": 7}
    exec(compile(tree, "<archived-analysis>", "exec"), namespace)
    assert namespace["a"] == Path("repo/outputs/v05_ActiveProgress奖励/run/metrics.csv")
    assert namespace["b"] == "outputs/v05_ActiveProgress奖励/run/final.pt"
    assert namespace["value"] == 0.99 * 7 + 100
    assert source == original
