"""Reproducible position perturbations without changing the physical task."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from envs import PursuitEnv
from scripts.evaluate_attacker_position_generalization import (
    episode_environment_config, sample_initial_position,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("margin", [0.0, 2.0, 5.0, 10.0])
def test_xyz_offsets_match_independent_seeded_uniform_draws(margin):
    center = np.array([497.49371855331, 0.0, 50.0])
    expected = center + np.random.default_rng(2031).uniform(-margin, margin, size=3)
    actual = sample_initial_position(center, margin, 2031)
    assert actual.shape == (3,) and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, sample_initial_position(center, margin, 2031))
    assert np.all(np.abs(actual - center) <= margin)
    np.testing.assert_array_equal(center, [497.49371855331, 0.0, 50.0])


def test_different_episode_seeds_change_the_position():
    center = [497.49371855331, 0.0, 50.0]
    assert not np.array_equal(sample_initial_position(center, 10.0, 4000),
                              sample_initial_position(center, 10.0, 4001))


def test_position_perturbation_is_not_projected_to_detection_sphere():
    center = np.array([497.49371855331, 0.0, 50.0])
    position = sample_initial_position(center, 10.0, 4000)
    assert not np.isclose(np.linalg.norm(position), np.linalg.norm(center), rtol=0, atol=1e-9)


@pytest.mark.parametrize("center,margin", [
    ([0.0, 0.0], 2.0),
    ([0.0, np.nan, 0.0], 2.0),
    ([0.0, 0.0, np.inf], 2.0),
    ([0.0, 0.0, 0.0], -2.0),
    ([0.0, 0.0, 0.0], np.inf),
])
def test_invalid_position_sampling_fails(center, margin):
    with pytest.raises(ValueError):
        sample_initial_position(center, margin, 4000)


def test_episode_config_changes_only_attacker_position_and_has_no_aliases():
    baseline = yaml.safe_load((ROOT / "configs/baseline_capture500.yaml").read_text(encoding="utf-8"))
    original = deepcopy(baseline)
    position = np.array([501.0, -3.0, 54.0])
    changed = episode_environment_config(baseline, position)
    expected = deepcopy(original)
    expected["initialization"]["attacker_position"] = position.tolist()
    assert changed == expected
    assert baseline == original
    changed["initialization"]["pursuer_positions"][0][0] += 20
    position[0] += 20
    assert baseline == original
    assert changed["initialization"]["attacker_position"] == [501.0, -3.0, 54.0]


def test_perturbed_episode_reset_preserves_velocities_and_clears_history(tmp_path):
    baseline = yaml.safe_load((ROOT / "configs/baseline_capture500.yaml").read_text(encoding="utf-8"))
    position = sample_initial_position(baseline["initialization"]["attacker_position"], 10.0, 4000)
    config = episode_environment_config(baseline, position)
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = PursuitEnv(path)
    observations, _ = env.reset(seed=4000)
    expected = np.array(baseline["initialization"]["pursuer_positions"])
    np.testing.assert_array_equal(env.states[:3, :3], expected)
    np.testing.assert_array_equal(env.states[-1, :3], position)
    np.testing.assert_array_equal(env.states[:, 3:], np.zeros((4, 3)))
    np.testing.assert_array_equal(env.accelerations, np.zeros((4, 3)))
    np.testing.assert_allclose(observations[:, 3:6], position - expected)
    after, rewards, _, _, _ = env.step(np.zeros((3, 3)))
    assert np.isfinite(after).all() and np.isfinite(rewards).all()
    env.reset(seed=4001)
    np.testing.assert_array_equal(env.states[-1, :3], position)
    np.testing.assert_array_equal(env.states[:, 3:], np.zeros((4, 3)))
    np.testing.assert_array_equal(env.accelerations, np.zeros((4, 3)))
