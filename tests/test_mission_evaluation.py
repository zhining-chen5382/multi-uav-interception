"""Task outcomes survive evaluation, CSV logging, and terminal bootstrapping."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithms.mappo import RolloutBuffer
from envs import PursuitEnv
from scripts.train_mappo import train
from utils.evaluation import run_episode, summarize_episodes


ROOT = Path(__file__).resolve().parents[1]


class ZeroActionAgent:
    def act(self, observations, deterministic=False):
        assert deterministic
        assert observations.shape == (3, 21)
        return np.zeros((3, 3)), None, None, None


def _mission_config(tmp_path, outcome):
    with (ROOT / "configs" / "baseline_3a1c.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["environment"]["max_episode_steps"] = 1
    initial = config["initialization"]
    initial["pursuer_velocities"] = [[0.0, 0.0, 0.0]] * 3
    initial["attacker_velocity"] = [0.0, 0.0, 0.0]
    initial["pursuer_positions"] = [[-30.0, 0.0, 0.0], [-30.0, 10.0, 0.0], [-30.0, -10.0, 0.0]]
    initial["attacker_position"] = [9.0, 0.0, 0.0] if outcome == "enemy" else [100.0, 0.0, 0.0]
    if outcome == "interception":
        initial["pursuer_positions"][0] = [96.0, 0.0, 0.0]
    path = tmp_path / (outcome + ".yaml")
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream)
    return path


@pytest.mark.parametrize("outcome,title", [
    ("interception", "Interception Success"),
    ("enemy", "Enemy Attack Success"),
    ("timeout", "Timeout"),
])
def test_evaluation_preserves_three_task_outcomes(tmp_path, outcome, title):
    env = PursuitEnv(_mission_config(tmp_path, outcome))
    metrics, trajectory = run_episode(env, ZeroActionAgent(), seed=2)
    assert metrics["outcome"] == title
    assert metrics["episode_length"] == 1
    assert metrics["captured"] == (outcome == "interception")
    assert metrics["interception_success"] == (outcome == "interception")
    assert metrics["enemy_attack_success"] == (outcome == "enemy")
    assert metrics["attacker_reached_base"] == (outcome == "enemy")
    assert metrics["truncated"] == (outcome == "timeout")
    assert metrics["capture_time"] == (env.dt if outcome == "interception" else None)
    assert trajectory["positions"].shape == (2, 4, 3)
    np.testing.assert_allclose(trajectory["distance_to_base"][-1], metrics["attacker_base_terminal_distance"])
    assert np.isfinite(trajectory["states"]).all()
    component_names = trajectory["reward_component_names"].tolist()
    components = trajectory["reward_components"]
    assert components.shape == (1, len(component_names))
    assert np.isfinite(components).all()
    weighted_names = ("active_progress_reward", "survival", "collision", "capture", "attack_failure")
    weighted_indices = [component_names.index(name) for name in weighted_names]
    np.testing.assert_allclose(components[:, weighted_indices].sum(axis=1), trajectory["rewards"])
    assert metrics["cumulative_active_progress_reward"] == pytest.approx(
        components[:, component_names.index("active_progress_reward")].sum()
    )
    assert metrics["cumulative_survival_penalty"] == pytest.approx(
        components[:, component_names.index("survival")].sum()
    )
    assert metrics["cumulative_collision_penalty"] == pytest.approx(
        components[:, component_names.index("collision")].sum()
    )
    assert metrics["terminal_reward"] == pytest.approx(
        components[:, [component_names.index("capture"),
                       component_names.index("attack_failure")]].sum()
    )
    assert sum(metrics[name] for name in (
        "cumulative_active_progress_reward", "cumulative_survival_penalty",
        "cumulative_collision_penalty", "terminal_reward",
    )) == pytest.approx(metrics["episode_reward"])
    summary = summarize_episodes([metrics])
    assert summary["interception_success_rate"] == float(outcome == "interception")
    assert summary["enemy_attack_success_rate"] == float(outcome == "enemy")
    assert summary["timeout_rate"] == float(outcome == "timeout")
    assert summary["average_interception_time"] == metrics["capture_time"]
    json.dumps(summary, allow_nan=False)


def test_enemy_terminal_is_logged_as_failure_and_does_not_bootstrap(tmp_path):
    environment_path = _mission_config(tmp_path, "enemy")
    with (ROOT / "configs" / "mappo_baseline.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["env_config"] = str(environment_path)
    config["training"]["rollout_steps"] = 2
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 2
    config_path = tmp_path / "training.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream)
    output = tmp_path / "run"
    summary = train(config_path, total_steps=2, output_dir=output)
    assert summary["episodes"] == 2
    with (output / "training_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["captured"] == "0" and row["enemy_attack_success"] == "1" for row in rows)
    assert all(row["outcome"] == "Enemy Attack Success" and row["capture_time"] == "" for row in rows)
    for row in rows:
        assert float(row["terminal_reward"]) == -100.0
        assert float(row["cumulative_survival_penalty"]) == pytest.approx(-0.02)
        assert sum(float(row[name]) for name in (
            "cumulative_active_progress_reward", "cumulative_survival_penalty",
            "cumulative_collision_penalty", "terminal_reward",
        )) == pytest.approx(float(row["episode_reward"]))
    with (output / "updates.csv").open(encoding="utf-8", newline="") as stream:
        update = next(csv.DictReader(stream))
    assert float(update["capture_rate"]) == 0.0
    assert float(update["enemy_attack_success_rate"]) == 1.0
    assert float(update["timeout_rate"]) == 0.0

    # The same terminal flag used by enemy victory suppresses the critic's next
    # state even if a large next value is accidentally supplied to the buffer.
    buffer = RolloutBuffer(1, 3, 21, 24, 3)
    buffer.add(np.zeros((3, 21)), np.zeros(24), np.zeros((3, 3)),
               np.zeros(3), -100.0, 3.0, 999.0, True, False)
    _, returns = buffer.compute_returns(gamma=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(returns, [-100.0])
