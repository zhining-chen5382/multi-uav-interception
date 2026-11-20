"""Acceptance checks for the independent trainer, without formal experiments.

The normalization numbers below are explicit unit-test fixtures. They do not
confirm or replace the physical-scale table required by the formal protocol.
"""

from copy import deepcopy
import csv
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from utils import formal_ppo_training as training


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCALES = [20, 20, 8, 3, 3, 2, 1500, 1500, 1500, 40, 40, 14, 750, 750, 750, 1]


@pytest.fixture
def config():
    result = training.load_configuration(ROOT / "configs/ppo_formal_v1.yaml")
    result["environment_config"]["normalization"].update(
        confirmed=True, scales=FIXTURE_SCALES, acceleration_kind="applied",
        confirmation_source="Unit-test fixture only")
    result["environment_config"]["environment"]["max_episode_steps"] = 7
    result["training"].update(total_env_steps=24, num_envs=2, rollout_steps=4,
                              evaluation_interval_steps=6, validation_episodes=5)
    result["network"].update(actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8])
    result["ppo"].update(update_epochs=1, minibatch_size=4, target_kl=1000.)
    result["experiment"]["seeds"] = [1001]
    return result


def evaluation_fixture(agent, environment_config, angles, seed, output_dir):
    """A cheap validation result for exercising trainer state transitions."""
    return dict(capture_rate=.8, defense_failure_rate=.2, timeout_rate=0.,
                average_interception_time=17., average_min_distance=4.2,
                average_return=100.)


def noisy_evaluation_fixture(agent, environment_config, angles, seed, output_dir):
    random.random()
    np.random.random(31)
    torch.randn(17)
    return evaluation_fixture(agent, environment_config, angles, seed, output_dir)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_same(first, second)
    else:
        assert left == right


def assert_same_training(left, right):
    left_agent, right_agent = deepcopy(left["agent"]), deepcopy(right["agent"])
    for agent in (left_agent, right_agent):
        agent["config"].pop("output_dir", None)
    assert_same(left_agent, right_agent)
    assert_same(left["training_state"], right["training_state"])
    assert_same(left["rng_state"], right["rng_state"])


def test_exact_evaluation_schedule_preserves_pending_rollout(config, tmp_path):
    calls = []

    def evaluate(agent, env_config, angles, seed, path):
        calls.append((angles.copy(), seed))
        return evaluation_fixture(agent, env_config, angles, seed, path)

    output = tmp_path / "schedule"
    summary = training.train(config, output, 1001, evaluation_fn=evaluate)
    rows = read_csv(output / "validation.csv")
    assert [int(row["env_steps"]) for row in rows] == [6, 12, 18, 24]
    assert [int(row["last_update_env_steps"]) for row in rows] == [0, 8, 16, 24]
    assert [int(row["pending_rollout_samples"]) for row in rows] == [6, 4, 2, 0]
    assert summary["status"] == "completed"
    assert summary["updates"] == 3
    assert summary["first_stable_env_steps"] == 18
    assert summary["env_steps"] == 24
    assert summary["best_env_steps"] == 6
    assert len(calls) == 4
    for angles, seed in calls:
        np.testing.assert_array_equal(angles, np.linspace(-30, 30, 5))
        assert seed == config["experiment"]["validation_seed"]
    best = load_checkpoint(output / "checkpoints/best.pt")
    assert best["training_state"]["env_steps"] == 6
    assert best["training_state"]["rollout_buffer"]["size"] == 3
    final = load_checkpoint(output / "checkpoints/final.pt")
    assert final["agent"]["scheduler"]["environment_steps"] == 24
    assert final["training_state"]["rollout_buffer"]["size"] == 0


@pytest.mark.parametrize("source_name", ["paused_000010.pt", "evaluated_000006.pt", "best.pt"])
def test_partial_rollout_checkpoint_restores_bitwise(config, tmp_path, source_name):
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    resumed = tmp_path / "resumed"
    training.train(config, uninterrupted, 1001, evaluation_fn=noisy_evaluation_fixture)
    summary = training.train(config, interrupted, 1001, stop_at_steps=10,
                             evaluation_fn=noisy_evaluation_fixture)
    assert summary["status"] == "paused"
    assert summary["pending_rollout_samples"] == 2
    source = interrupted / "checkpoints" / source_name
    saved = load_checkpoint(source)
    assert saved["training_state"]["rollout_buffer"]["size"] > 0
    summary = training.train(config, resumed, 1001, resume_checkpoint=source,
                             evaluation_fn=noisy_evaluation_fixture)
    assert summary["status"] == "completed"
    expected = load_checkpoint(uninterrupted / "checkpoints/final.pt")
    actual = load_checkpoint(resumed / "checkpoints/final.pt")
    assert_same_training(expected, actual)
    for name in training.LOG_FIELDS:
        assert read_csv(uninterrupted / (name + ".csv")) == read_csv(resumed / (name + ".csv"))
    assert (resumed / "checkpoints/initial.pt").is_file()
    assert (resumed / "checkpoints/best.pt").is_file()
    assert (resumed / "checkpoints/resume_initial.pt").is_file()


def test_resume_at_coincident_update_validation_boundary(config, tmp_path):
    config["training"]["evaluation_interval_steps"] = 8
    original, resumed = tmp_path / "original", tmp_path / "resumed"
    training.train(config, original, 1001, evaluation_fn=evaluation_fixture)
    checkpoint = original / "checkpoints/checkpoint_000008.pt"
    saved = load_checkpoint(checkpoint)
    assert saved["training_state"]["next_evaluation_steps"] == 16
    assert saved["training_state"]["logs"]["validation"][-1]["env_steps"] == 8
    training.train(config, resumed, 1001, resume_checkpoint=checkpoint,
                   evaluation_fn=evaluation_fixture)
    assert_same_training(load_checkpoint(original / "checkpoints/final.pt"),
                         load_checkpoint(resumed / "checkpoints/final.pt"))
    assert [int(row["env_steps"]) for row in read_csv(resumed / "validation.csv")] == [8, 16, 24]


def test_validation_global_rng_isolation(config, tmp_path):
    quiet, noisy = tmp_path / "quiet", tmp_path / "noisy"
    training.train(config, quiet, 1001, evaluation_fn=evaluation_fixture)
    training.train(config, noisy, 1001, evaluation_fn=noisy_evaluation_fixture)
    assert_same_training(load_checkpoint(quiet / "checkpoints/final.pt"),
                         load_checkpoint(noisy / "checkpoints/final.pt"))


def test_validation_refuses_model_mutation(config, tmp_path):
    def invalid_evaluation(agent, env_config, angles, seed, path):
        with torch.no_grad():
            agent.actor.log_std.add_(.1)
        return evaluation_fixture(agent, env_config, angles, seed, path)

    with pytest.raises(ValueError, match="Validation changed"):
        training.train(config, tmp_path / "mutated", 1001, evaluation_fn=invalid_evaluation)
    assert not (tmp_path / "mutated/checkpoints/final.pt").exists()


def test_eight_environment_states_and_rng_are_independent(config):
    vector = training.IndependentEnvironments(config["environment_config"], 8, 1001)
    observations, _ = vector.reset()
    initial = deepcopy(vector.state_dict())
    assert observations.shape == (8, 16)
    assert len(set(vector.seeds)) == 8
    assert len({id(env.rng) for env in vector.envs}) == 8
    assert len({id(env.states) for env in vector.envs}) == 8
    assert len({env.angle_degrees for env in vector.envs}) == 8
    vector.envs[0].step(np.zeros(3))
    vector.envs[0].reset()
    for index in range(1, 8):
        assert_same(initial["environments"][index], vector.envs[index].get_state())
    repeated = training.IndependentEnvironments(config["environment_config"], 8, 1001)
    repeated.reset()
    assert_same(initial, repeated.state_dict())
    restored = training.IndependentEnvironments(config["environment_config"], 8, 1002)
    restored.reset()
    restored.load_state_dict(vector.state_dict())
    for _ in range(3):
        assert_same(vector.step(np.zeros((8, 3))), restored.step(np.zeros((8, 3))))


def test_tail_rollout_budget_and_callback(config, tmp_path):
    config["training"]["total_env_steps"] = 26
    config["training"]["validation_episodes"] = 3
    callbacks = []
    output = tmp_path / "tail"
    summary = training.train(config, output, 1001, evaluation_fn=evaluation_fixture,
                             post_training_fn=lambda *args: callbacks.append(args))
    assert summary["env_steps"] == 26
    assert [int(row["rollout_samples"]) for row in read_csv(output / "updates.csv")] == [8, 8, 8, 2]
    assert [int(row["env_steps"]) for row in read_csv(output / "validation.csv")] == [6, 12, 18, 24, 26]
    assert len(callbacks) == 1
    assert callbacks[0][0] == output.resolve()
    assert callbacks[0][1]["training"]["total_env_steps"] == 26
    assert callbacks[0][2]["status"] == "completed"


@pytest.mark.parametrize("change", ["unconfirmed", "missing_scales", "missing_acceleration", "misaligned_budget"])
def test_invalid_protocol_is_rejected_before_creating_output(config, tmp_path, change):
    if change == "unconfirmed":
        config["environment_config"]["normalization"]["confirmed"] = False
    elif change == "missing_scales":
        config["environment_config"]["normalization"]["scales"] = None
    elif change == "missing_acceleration":
        config["environment_config"]["normalization"]["acceleration_kind"] = None
    else:
        config["training"]["total_env_steps"] = 25
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        training.train(config, output, 1001, evaluation_fn=evaluation_fixture)
    assert not output.exists()


def test_existing_output_and_checkpoint_are_preserved(config, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new output directory"):
        training.train(config, output, 1001, evaluation_fn=evaluation_fixture)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize("change", ["budget", "seed", "environment"])
def test_exact_resume_rejects_changed_protocol(config, tmp_path, change):
    source = tmp_path / "source"
    training.train(config, source, 1001, stop_at_steps=10, evaluation_fn=evaluation_fixture)
    checkpoint = source / "checkpoints/paused_000010.pt"
    original = checkpoint.read_bytes()
    seed = 1001
    if change == "budget":
        config["training"]["total_env_steps"] = 26
    elif change == "seed":
        seed = 1002
    else:
        config["environment_config"]["normalization"]["scales"] = [1000.] * 16
    with pytest.raises(ValueError, match="Resume changes"):
        training.train(config, tmp_path / "rejected", seed, resume_checkpoint=checkpoint,
                       evaluation_fn=evaluation_fixture)
    assert not (tmp_path / "rejected").exists()
    assert checkpoint.read_bytes() == original


class CounterEnvironment:
    """An inexpensive environment fixture for the full aggregate-counter test."""

    transition_count = 0

    def __init__(self, config, seed):
        self.config = deepcopy(config)
        self.rng = np.random.default_rng(seed)
        self.dt = config["environment"]["dt"]
        self.max_episode_steps = config["environment"]["max_episode_steps"]

    def _info(self):
        return dict(min_distance=100., distance_to_base=200., captured=False,
                    enemy_attack_success=False, timeout=self.step_count == self.max_episode_steps,
                    normalization_violations=[])

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.angle_degrees = float(self.rng.uniform(-30, 30))
        self.states = np.zeros((2, 6))
        self.states[1, 0] = 100
        self.step_count = 0
        return np.zeros(16, dtype=np.float32), self._info()

    def step(self, action):
        self.step_count += 1
        CounterEnvironment.transition_count += 1
        return np.zeros(16, dtype=np.float32), 0., self.step_count == self.max_episode_steps, False, self._info()

    def get_state(self):
        return dict(states=self.states.copy(), step_count=self.step_count, angle_degrees=self.angle_degrees,
                    rng=deepcopy(self.rng.bit_generator.state))

    def set_state(self, state):
        self.states = state["states"].copy()
        self.step_count = state["step_count"]
        self.angle_degrees = state["angle_degrees"]
        self.rng.bit_generator.state = deepcopy(state["rng"])


def test_actual_300k_counter_and_exact_10k_milestones(config, tmp_path, monkeypatch):
    """Exercise all 300k collection counters with cheap dynamics/update fixtures.

    Real dynamics and PPO numerical correctness are covered by the smaller
    integration tests and their own suites. This check tests the production
    8x512 budget, cadence, final short rollout and scheduler horizon.
    """
    config["training"].update(total_env_steps=300000, num_envs=8, rollout_steps=512,
                              evaluation_interval_steps=10000, validation_episodes=100)
    config["environment_config"]["environment"]["max_episode_steps"] = 500
    CounterEnvironment.transition_count = 0
    monkeypatch.setattr(training, "InterceptionEnv", CounterEnvironment)

    def act(agent, observations):
        count = len(observations)
        actions = np.zeros((count, 3), dtype=np.float32)
        return dict(actions=actions, raw_actions=actions, log_probs=np.zeros(count),
                    means=actions, log_stds=np.full_like(actions, np.log(.5)), values=np.zeros(count))

    def update(agent, buffer, env_steps):
        assert buffer.returns_ready
        rates = agent.scheduler.step(env_steps)
        agent.update_count += 1
        return dict(actor_loss=0., value_loss=0., entropy=0., approx_kl=0., clip_fraction=0.,
                    actor_grad_norm=0., critic_grad_norm=0., actor_learning_rate=rates[0],
                    critic_learning_rate=rates[1], updates_completed_minibatches=1,
                    epochs_completed=1, early_stopped=False, minibatches=[], epochs=[])

    monkeypatch.setattr(training.PPOAgent, "act", act)
    monkeypatch.setattr(training.PPOAgent, "value", lambda agent, observations: np.zeros(len(observations)))
    monkeypatch.setattr(training.PPOAgent, "update", update)
    output = tmp_path / "counter_300k"
    calls = []

    def evaluate(agent, env_config, angles, seed, path):
        calls.append(angles.copy())
        return evaluation_fixture(agent, env_config, angles, seed, path)

    summary = training.train(config, output, 1001, evaluation_fn=evaluate)
    assert CounterEnvironment.transition_count == 300000
    assert summary["env_steps"] == 300000
    assert summary["updates"] == 74
    assert summary["first_stable_env_steps"] == 30000
    validation = read_csv(output / "validation.csv")
    assert [int(row["env_steps"]) for row in validation] == list(range(10000, 300001, 10000))
    assert len(calls) == 30
    for angles in calls:
        np.testing.assert_array_equal(angles, np.linspace(-30, 30, 100))
    updates = read_csv(output / "updates.csv")
    assert [int(row["rollout_samples"]) for row in updates] == [4096] * 73 + [992]
    assert sum(int(row["rollout_samples"]) for row in updates) == 300000
    assert float(updates[-1]["actor_learning_rate"]) == config["ppo"]["actor_final_learning_rate"]
    assert float(updates[-1]["critic_learning_rate"]) == config["ppo"]["critic_final_learning_rate"]
    final = load_checkpoint(output / "checkpoints/final.pt")
    assert final["training_state"]["rollout_buffer"]["size"] == 0
    assert len(final["training_state"]["vector_environments"]["environments"]) == 8
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"] == "completed"
