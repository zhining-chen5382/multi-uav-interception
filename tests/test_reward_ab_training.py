"""Reward-B logging and exact recovery using short engineering-only budgets."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.ppo import PPOAgent
from scripts.offline_best_so_far_reward_replay import best_so_far_shaping
from utils.formal_ppo_evaluation import evaluate_angles
from utils.formal_ppo_training import _state_hash, load_configuration, train


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    result = load_configuration(ROOT / "configs/ppo_formal_v1.yaml")
    result["environment_config"]["reward"]["best_so_far_shaping"] = dict(length_scale_m=20.0, beta=100.0)
    # Short budgets, small networks and early time limits are unit-test fixtures.
    result["environment_config"]["environment"]["max_episode_steps"] = 13
    result["training"].update(total_env_steps=48, num_envs=2, rollout_steps=8,
                              evaluation_interval_steps=12, validation_episodes=3)
    result["network"].update(actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8])
    result["ppo"].update(update_epochs=1, minibatch_size=8)
    return result


def evaluation_fixture(*args):
    return dict(capture_rate=0.0, defense_failure_rate=0.0, timeout_rate=1.0,
                average_interception_time=None, average_min_distance=400.0, average_return=-5.0)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_b_partial_rollout_resume_preserves_full_state_and_reward_logs(config, tmp_path):
    continuous, paused, restored = (tmp_path / name for name in ("continuous", "paused", "restored"))
    train(config, continuous, 1001, evaluation_fn=evaluation_fixture)
    summary = train(config, paused, 1001, stop_at_steps=20, evaluation_fn=evaluation_fixture)
    assert summary["pending_rollout_samples"] == 4
    checkpoint = paused / "checkpoints/paused_000020.pt"
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for environment in bundle["training_state"]["vector_environments"]["environments"]:
        assert environment["step_count"] == 10
        assert environment["best_distance_so_far"] == environment["episode_min_distance"]
        assert "near_shaping" in environment["episode_reward_components"]
    train(config, restored, 1001, resume_checkpoint=checkpoint, evaluation_fn=evaluation_fixture)
    left = torch.load(continuous / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    right = torch.load(restored / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    for state in (left, right):
        state["agent"]["config"].pop("output_dir", None)
    for key in ("agent", "training_state", "rng_state"):
        assert _state_hash(left[key]) == _state_hash(right[key])
    for filename in continuous.glob("*.csv"):
        assert filename.read_bytes() == (restored / filename.name).read_bytes()
    rows = read_csv(continuous / "episodes.csv")
    assert rows and "near_shaping_return" in rows[0]
    for row in rows:
        assert float(row["near_shaping_return"]) >= 0
        assert float(row["episode_return"]) == pytest.approx(
            float(row["original_return"]) + float(row["near_shaping_return"]), abs=1e-10)


def test_b_evaluation_saves_actual_best_distance_and_shaping_arrays(config, tmp_path):
    agent = PPOAgent(config)
    output = tmp_path / "evaluation"
    evaluate_angles(agent, config["environment_config"], [-20.0, 20.0], 2001, output)
    for row in read_csv(output / "episodes.csv"):
        with np.load(output / row["trajectory"], allow_pickle=False) as trajectory:
            states = trajectory["states"]
            distances = np.linalg.norm(states[:, 0, :3] - states[:, 1, :3], axis=1)
            np.testing.assert_array_equal(trajectory["best_distances"], np.minimum.accumulate(distances))
            np.testing.assert_array_equal(trajectory["near_shaping"], best_so_far_shaping(distances, 20.0, 100.0))
            assert float(row["near_shaping_return"]) == pytest.approx(trajectory["near_shaping"].sum(), abs=1e-10)
            assert float(row["original_return"]) + float(row["near_shaping_return"]) == pytest.approx(
                trajectory["rewards"].sum(), abs=1e-10)
            metadata = json.loads(trajectory["metadata_json"].item())
            assert metadata["best_distance_so_far"] == float(distances.min())


def test_a_episode_schema_and_reward_arrays_remain_compatible(config, tmp_path):
    original = deepcopy(config)
    original["environment_config"]["reward"].pop("best_so_far_shaping")
    run = tmp_path / "a_control"
    train(original, run, 1001, evaluation_fn=evaluation_fixture)
    assert "near_shaping_return" not in read_csv(run / "episodes.csv")[0]
    agent = PPOAgent(original)
    evaluate_angles(agent, original["environment_config"], [0.0], 2001, tmp_path / "a_evaluation")
    with np.load(tmp_path / "a_evaluation/trajectories/episode_000.npz") as trajectory:
        assert "near_shaping" not in trajectory.files
        assert "best_distances" not in trajectory.files
