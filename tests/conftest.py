"""Shared configuration fixtures for the baseline environment tests."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def baseline_config():
    with (ROOT / "configs" / "baseline_3a1c.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def config_factory(tmp_path, baseline_config):
    """Write a separate YAML file with per-section overrides for each test."""
    counter = 0

    def make_config(**overrides):
        nonlocal counter
        config = deepcopy(baseline_config)
        for section, values in overrides.items():
            config[section].update(values)
        path = tmp_path / ("config_%d.yaml" % counter)
        counter += 1
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        return path

    return make_config
