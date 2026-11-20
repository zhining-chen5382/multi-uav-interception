"""Independent cross-module review checks using explicit engineering fixtures.

These short checks exercise production collector/model code. Synthetic
normalization divisors remain local to the tests and do not approve the
unconfirmed formal observation contract.
"""

from copy import deepcopy
import csv
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from algorithms.ppo import PPOAgent
from utils import formal_ppo_training as training
from utils.formal_ppo_evaluation import evaluate_angles, evaluation_context


ROOT = Path(__file__).resolve().parents[1]


def engineering_config():
    config = training.load_configuration(ROOT / "configs/ppo_formal_v1.yaml")
    config["environment_config"]["normalization"].update(
        confirmed=True, scales=[1000.] * 16, acceleration_kind="applied",
        confirmation_source="Independent integration test fixture only")
    config["environment_config"]["environment"]["max_episode_steps"] = 5
    config["training"].update(total_env_steps=104, num_envs=8, rollout_steps=512,
                              evaluation_interval_steps=40, validation_episodes=3)
    config["ppo"].update(update_epochs=2, minibatch_size=16)
    return config


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


def checkpoint_learning_state(bundle):
    result = deepcopy({key: bundle[key] for key in ("agent", "training_state", "rng_state")})
    result["agent"]["config"].pop("output_dir", None)
    return result


def test_eight_environment_real_evaluation_and_partial_buffer_resume(tmp_path):
    """A real eight-env collector crosses episodes with a nonempty 512 buffer."""
    config = engineering_config()
    original = tmp_path / "original"
    paused = tmp_path / "paused"
    resumed = tmp_path / "resumed"
    training.train(config, original, 1001, evaluation_fn=evaluate_angles)
    training.train(config, paused, 1001, stop_at_steps=56, evaluation_fn=evaluate_angles)
    source = paused / "checkpoints/paused_000056.pt"
    source_bytes = source.read_bytes()
    saved = checkpoint(source)
    assert saved["training_state"]["rollout_buffer"]["size"] == 7
    assert saved["training_state"]["rollout_buffer"]["capacity"] == 512
    assert saved["training_state"]["episodes"] == 8
    assert saved["training_state"]["next_evaluation_steps"] == 80
    training.train(config, resumed, 1001, resume_checkpoint=source, evaluation_fn=evaluate_angles)
    expected = checkpoint(original / "checkpoints/final.pt")
    actual = checkpoint(resumed / "checkpoints/final.pt")
    assert_equal(checkpoint_learning_state(expected), checkpoint_learning_state(actual))
    assert source.read_bytes() == source_bytes
    assert actual["training_state"]["env_steps"] == 104
    assert actual["agent"]["scheduler"]["environment_steps"] == 104
    assert actual["agent"]["actor_optimizer"]["state"]
    assert actual["agent"]["critic_optimizer"]["state"]
    assert actual["training_state"]["rollout_buffer"]["size"] == 0
    with (resumed / "validation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["env_steps"]) for row in rows] == [40, 80, 104]
    for filename in ("episodes.csv", "updates.csv", "minibatches.csv", "epochs.csv", "validation.csv"):
        assert (original / filename).read_bytes() == (resumed / filename).read_bytes()


def test_best_checkpoint_uses_all_four_priority_levels(tmp_path):
    config = engineering_config()
    config["training"].update(total_env_steps=80, evaluation_interval_steps=8)
    # Each tuple is capture rate, failure rate, successful capture time, min D.
    # Later rows exercise exact ties and regressions at each priority level.
    outcomes = iter([
        (.5, .2, 12., 4.),
        (.4, .1, 10., 3.),
        (.5, .1, 12., 4.),
        (.5, .1, 10., 4.),
        (.5, .1, 10., 3.),
        (.5, .1, 10., 3.),
        (.4, .0, 1., .1),
        (.5, .2, 1., .1),
        (.5, .1, 11., .1),
        (.5, .1, 10., 4.),
    ])

    def evaluate(agent, environment, angles, seed, output):
        capture, failure, time, distance = next(outcomes)
        return dict(capture_rate=capture, defense_failure_rate=failure,
                    timeout_rate=1-capture-failure, average_interception_time=time,
                    average_min_distance=distance, average_return=0.)

    output = tmp_path / "priority"
    summary = training.train(config, output, 1001, evaluation_fn=evaluate)
    assert summary["best_env_steps"] == 40
    saved = checkpoint(output / "checkpoints/best.pt")
    assert saved["is_best_checkpoint"] is True
    assert saved["training_state"]["env_steps"] == 40
    assert saved["training_state"]["best_key"] == (.5, -.1, -10., -3.)
    with (output / "validation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["env_steps"]) for row in rows if row["new_best"] == "True"] == [8, 24, 32, 40]
    assert summary["env_steps"] == 80


def test_evaluation_context_restores_all_rng_after_inference_error():
    config = engineering_config()
    agent = PPOAgent(config)
    agent.actor.train()
    agent.critic.eval()
    before = training.global_rng_state()
    weights = deepcopy(agent.state_dict())
    with pytest.raises(RuntimeError, match="synthetic inference interruption"):
        with evaluation_context(agent):
            assert not agent.actor.training and not agent.critic.training
            assert torch.is_inference_mode_enabled()
            random.random()
            np.random.normal(size=23)
            torch.randn(29)
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    torch.randn(3, device="cuda:{}".format(index))
            raise RuntimeError("synthetic inference interruption")
    assert_equal(before, training.global_rng_state())
    assert_equal(weights, agent.state_dict())
    assert agent.actor.training and not agent.critic.training
