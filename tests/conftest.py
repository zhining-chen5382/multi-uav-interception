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
def legacy_config(baseline_config):
    """Explicit old pursuit/boundary scenario, independent of experiment outputs.

    Legacy policy and clipping coverage must survive changes to the default
    task. Physical limits still come from the baseline, which keeps them fixed.
    """
    config = deepcopy(baseline_config)
    config.pop("mission", None)
    aircraft = config.pop("attacker", config.get("evader"))
    config["evader"] = deepcopy(aircraft)
    config["evader"]["policy"] = {
        "distance_epsilon": 1e-6,
        "boundary_margin_horizontal": 50.0,
        "boundary_margin_vertical": 20.0,
        "boundary_strength": 2.0,
    }
    config["environment"]["bounds"] = {
        "low": [-300.0, -300.0, 0.0], "high": [300.0, 300.0, 150.0],
    }
    config["initialization"] = {
        "pursuer_positions": [[-30.0, -10.0, 50.0], [-30.0, 0.0, 50.0],
                              [-30.0, 10.0, 50.0]],
        "pursuer_velocities": [[0.0, 0.0, 0.0]] * 3,
        "evader_position": [0.0, 0.0, 50.0],
        "evader_velocity": [0.0, 0.0, 0.0],
    }
    config["reward"].pop("attack_failure_penalty", None)
    # Historical pursuit snapshots use the old, mutually exclusive reward
    # schema; deriving other physical values from the default must not silently
    # migrate those regression scenarios to point-defense shaping.
    config["reward"].pop("active_progress_coefficient", None)
    config["reward"].pop("survival_penalty_per_second", None)
    config["reward"].update(progress_coefficient=1.0, time_penalty=-0.01)
    return config


def _configuration_writer(tmp_path, source, prefix):
    counter = 0

    def make_config(**overrides):
        nonlocal counter
        config = deepcopy(source)
        for section, values in overrides.items():
            config[section].update(values)
        path = tmp_path / ("%s_%d.yaml" % (prefix, counter))
        counter += 1
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        return path

    return make_config


@pytest.fixture
def config_factory(tmp_path, baseline_config):
    """Write current-task YAML files with per-section overrides."""
    return _configuration_writer(tmp_path, baseline_config, "config")


@pytest.fixture
def legacy_config_factory(tmp_path, legacy_config):
    """Write isolated historical pursuit configs for retained regression tests."""
    return _configuration_writer(tmp_path, legacy_config, "legacy_config")
