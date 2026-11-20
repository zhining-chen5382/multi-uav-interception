"""Ensure evaluation control logging is passive and physically unambiguous."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from envs import PursuitEnv
from scripts.evaluate_direction_generalization import ControlRecordingEnv, record_episode
from utils.evaluation import run_episode


ROOT = Path(__file__).resolve().parents[1]


class ConstantAgent:
    def __init__(self, action):
        self.action = np.tile(action, (3, 1))

    def act(self, observations, deterministic=False):
        assert deterministic and observations.shape == (3, 21)
        return self.action.copy(), None, None, None


def environment(tmp_path, velocity=0.0):
    config = yaml.safe_load((ROOT / "configs/baseline_capture500.yaml").read_text(encoding="utf-8"))
    config["environment"]["max_episode_steps"] = 2
    config["initialization"]["pursuer_velocities"] = [[velocity, 0.0, 0.0]] * 3
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return PursuitEnv(path)


def test_control_recording_does_not_change_trajectory_or_reward(tmp_path):
    agent = ConstantAgent([0.7, -0.2, 0.1])
    expected_metrics, expected = run_episode(environment(tmp_path), agent, 1000)
    actual_metrics, actual = record_episode(environment(tmp_path), agent, 1000)
    assert actual_metrics == expected_metrics
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["observations"].shape == (2, 3, 21)
    assert actual["aircraft_accelerations"].shape == (3, 4, 3)
    for key in ("command_accelerations", "applied_accelerations", "realized_accelerations"):
        assert actual[key].shape == (2, 3, 3)
        assert np.isfinite(actual[key]).all()


def test_scaled_command_and_jerk_limited_acceleration_are_distinct(tmp_path):
    _, trajectory = record_episode(environment(tmp_path), ConstantAgent([1.0, 1.0, 1.0]), 1000)
    expected = np.tile([3 / np.sqrt(2), 3 / np.sqrt(2), 2.0], (3, 1))
    np.testing.assert_allclose(trajectory["command_accelerations"][0], expected)
    np.testing.assert_allclose(np.linalg.norm(trajectory["applied_accelerations"][0], axis=-1), 0.5)
    np.testing.assert_allclose(trajectory["realized_accelerations"][0], trajectory["applied_accelerations"][0])
    assert np.linalg.norm(trajectory["command_accelerations"][0, 0]) > 0.5


def test_realized_acceleration_reflects_speed_clipping(tmp_path):
    _, trajectory = record_episode(environment(tmp_path, velocity=19.99), ConstantAgent([1.0, 0.0, 0.0]), 1000)
    assert trajectory["applied_accelerations"][0, 0, 0] == pytest.approx(0.5)
    assert trajectory["realized_accelerations"][0, 0, 0] == pytest.approx(0.1)
    np.testing.assert_allclose(trajectory["states"][1, :3, 3], 20.0)


def test_reset_clears_control_and_acceleration_histories(tmp_path):
    recorder = ControlRecordingEnv(environment(tmp_path))
    before, _ = recorder.reset(seed=1000)
    recorder.step(np.ones((3, 3)))
    assert len(recorder.command_history) == 1
    after, _ = recorder.reset(seed=1001)
    np.testing.assert_array_equal(after, before)
    assert recorder.command_history == recorder.applied_history == recorder.realized_history == []
    assert recorder.observations_history == []
    assert len(recorder.aircraft_acceleration_history) == 1
    np.testing.assert_array_equal(recorder.aircraft_acceleration_history[0], np.zeros((4, 3)))
