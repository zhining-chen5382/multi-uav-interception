"""Production contracts for the independent interception-line reward experiment."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.ppo import PPOAgent
from envs.interception_env import InterceptionEnv, REWARD_FIELDS
from utils.formal_ppo_evaluation import evaluate_angles
from utils.formal_ppo_training import _state_hash, load_configuration, train
from utils.line_cost_replay import line_cost, segment_distances


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base_config():
    return yaml.safe_load((ROOT / "configs/interception_1v1.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def line_config(base_config):
    result = deepcopy(base_config)
    result["reward"].update(line_cost_coefficient=0.1,
                            best_so_far_shaping=dict(length_scale_m=20.0, beta=100.0))
    return result


def stationary_environment(config):
    environment = InterceptionEnv(config, seed=1001)
    environment.reset(angle_degrees=0.0)
    environment.attacker_policy = lambda state: np.zeros(3)
    return environment


def place_aircraft(environment, pursuer, attacker):
    """Isolate reward geometry using zero motion in one controlled transition."""
    environment.states[:, :3] = [pursuer, attacker]
    environment.states[:, 3:] = 0.0
    environment.accelerations[:] = 0.0


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("pursuer,attacker,expected", [
    ([50, 0, 0], [100, 0, 0], 0.0),
    ([50, 3, 4], [100, 0, 0], 5.0),
    ([50, -3, -4], [100, 0, 0], 5.0),
    ([50, 0, 6], [100, 0, 0], 6.0),
    ([-3, 4, 0], [100, 0, 0], 5.0),
    ([103, 4, 0], [100, 0, 0], 5.0),
    ([5, 1, 1], [10, 2, 2], 0.0),
    ([10, 0, 0], [0, 0, 0], 10.0),
    ([0, 0, 0], [0, 0, 0], 0.0),
])
def test_line_cost_uses_finite_closed_segment_3d_geometry(line_config, pursuer, attacker, expected):
    environment = stationary_environment(line_config)
    place_aircraft(environment, pursuer, attacker)

    observation, reward, _, truncated, info = environment.step(np.zeros(3))

    assert observation.shape == (16,)
    assert np.isfinite(observation).all() and np.isfinite(reward)
    assert not truncated
    assert info["line_segment_distance"] == pytest.approx(expected, abs=1e-12)
    assert info["reward_components"]["line_cost"] == pytest.approx(-0.1 * expected * environment.dt)
    assert info["episode_reward_components"]["line_cost"] == info["reward_components"]["line_cost"]
    assert environment.reward_fields == REWARD_FIELDS + ("near_shaping", "line_cost")
    assert reward == pytest.approx(sum(info["reward_components"].values()), abs=1e-12)


def test_cost_is_evaluated_after_dynamics_not_at_start_of_transition(line_config):
    environment = stationary_environment(line_config)
    place_aircraft(environment, [50, 2, 0], [100, 0, 0])
    environment.states[0, 4] = 1.0

    _, _, _, _, info = environment.step(np.zeros(3))

    assert environment.states[0, 1] == pytest.approx(2.1)
    assert info["line_segment_distance"] == pytest.approx(2.1)
    assert info["reward_components"]["line_cost"] == pytest.approx(-0.021)


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
def test_line_cost_scales_with_physical_time(line_config, dt):
    line_config["environment"]["dt"] = dt
    environment = stationary_environment(line_config)
    place_aircraft(environment, [50, 3, 4], [100, 0, 0])

    _, _, _, _, info = environment.step(np.zeros(3))

    assert info["reward_components"]["line_cost"] == pytest.approx(-0.1 * 5.0 * dt)


def test_explicit_zero_coefficient_keeps_diagnostic_without_cost(line_config):
    line_config["reward"]["line_cost_coefficient"] = 0.0
    environment = stationary_environment(line_config)
    place_aircraft(environment, [50, 3, 4], [100, 0, 0])

    _, _, _, _, info = environment.step(np.zeros(3))

    assert info["line_segment_distance"] == 5.0
    assert info["reward_components"]["line_cost"] == 0.0


def test_reset_clears_accumulated_cost_and_exposes_initial_geometry(line_config):
    environment = stationary_environment(line_config)
    for _ in range(3):
        place_aircraft(environment, [50, 3, 4], [100, 0, 0])
        environment.step(np.zeros(3))
    assert environment.get_state()["episode_reward_components"]["line_cost"] == pytest.approx(-0.15)

    observation, info = environment.reset(angle_degrees=20.0)

    assert observation.shape == (16,)
    assert info["reward_components"]["line_cost"] == 0.0
    assert info["episode_reward_components"]["line_cost"] == 0.0
    actual = segment_distances(environment.states[0, :3], environment.states[1, :3], environment.base_position)
    assert info["line_segment_distance"] == pytest.approx(float(actual))


@pytest.mark.parametrize("pursuer,attacker,captured,failed", [
    ([101, 2, 0], [100, 0, 0], True, False),
    ([8, 3, 0], [8, 0, 0], True, False),
    ([20, 3, 0], [8, 0, 0], False, True),
])
def test_terminal_step_retains_line_cost_and_original_terminal_rewards(
        line_config, pursuer, attacker, captured, failed):
    environment = stationary_environment(line_config)
    place_aircraft(environment, pursuer, attacker)

    _, reward, terminated, truncated, info = environment.step(np.zeros(3))

    assert terminated and not truncated
    assert info["captured"] == captured
    assert info["enemy_attack_success"] == failed
    components = info["reward_components"]
    assert components["capture"] == (500.0 if captured else 0.0)
    assert components["attack_failure"] == (-100.0 if failed else 0.0)
    assert components["line_cost"] < 0
    assert components["line_cost"] == pytest.approx(-0.1 * info["line_segment_distance"] * environment.dt)
    assert reward == pytest.approx(sum(components.values()), abs=1e-10)


def test_timeout_terminal_step_includes_line_cost(line_config):
    environment = stationary_environment(line_config)
    place_aircraft(environment, [50, 3, 4], [100, 0, 0])
    environment.step_count = environment.max_episode_steps - 1

    _, _, terminated, truncated, info = environment.step(np.zeros(3))

    assert terminated and not truncated and info["timeout"]
    assert info["reward_components"]["capture"] == 0.0
    assert info["reward_components"]["attack_failure"] == 0.0
    assert info["reward_components"]["line_cost"] == pytest.approx(-0.05)


def test_random_complete_episode_matches_offline_cost_at_every_step(line_config):
    environment = InterceptionEnv(line_config, seed=1001)
    _, initial = environment.reset(angle_degrees=-12.0)
    generator = np.random.default_rng(7913)
    geometry = [initial["line_segment_distance"]]
    times = [0.0]
    costs, rewards = [], []
    done = False
    while not done:
        observation, reward, done, truncated, info = environment.step(generator.uniform(-1, 1, 3))
        assert observation.shape == (16,) and np.isfinite(observation).all()
        assert np.isfinite(reward) and not truncated
        geometry.append(float(segment_distances(environment.states[0, :3], environment.states[1, :3],
                                               environment.base_position)))
        times.append(environment.step_count * environment.dt)
        assert info["line_segment_distance"] == pytest.approx(geometry[-1], abs=1e-12)
        costs.append(info["reward_components"]["line_cost"])
        rewards.append(reward)
    assert done
    np.testing.assert_allclose(costs, line_cost(geometry, times, 0.1), rtol=0, atol=1e-12)
    assert np.all(np.asarray(costs) <= 0)
    assert sum(costs) == pytest.approx(info["episode_reward_components"]["line_cost"], abs=1e-10)
    assert sum(rewards) == pytest.approx(environment.episode_return, abs=1e-10)


def test_snapshot_restores_accumulated_cost_and_future_steps(line_config):
    environment = InterceptionEnv(line_config, seed=1091)
    environment.reset(angle_degrees=12.0)
    actions = np.random.default_rng(1091).uniform(-1, 1, (20, 3))
    for action in actions[:10]:
        environment.step(action)
    snapshot = environment.get_state()
    assert snapshot["episode_reward_components"]["line_cost"] < 0
    clone = InterceptionEnv(line_config, seed=44)
    clone.set_state(snapshot)
    np.testing.assert_equal(clone.get_state(), snapshot)
    for action in actions[10:]:
        np.testing.assert_equal(environment.step(action), clone.step(action))
        np.testing.assert_equal(environment.get_state(), clone.get_state())
    np.testing.assert_equal(environment.reset(), clone.reset())


@pytest.mark.parametrize("container,corruption", [
    ("reward_components", "missing"), ("episode_reward_components", "missing"),
    ("reward_components", "extra"), ("episode_reward_components", "extra"),
    ("reward_components", np.nan), ("reward_components", np.inf),
    ("episode_reward_components", np.nan), ("episode_reward_components", -np.inf),
    ("reward_components", 1.0), ("episode_reward_components", 1.0),
    ("reward_components", True), ("episode_reward_components", True),
    ("reward_components", "0.0"), ("episode_reward_components", "0.0"),
])
def test_corrupt_line_components_are_rejected_atomically(line_config, container, corruption):
    environment = InterceptionEnv(line_config, seed=1001)
    environment.reset(angle_degrees=-3.0)
    environment.step([0.1, 0.2, -0.1])
    original = environment.get_state()
    invalid = deepcopy(original)
    if corruption == "missing":
        invalid[container].pop("line_cost")
    elif corruption == "extra":
        invalid[container]["unknown_reward"] = 0.0
    else:
        invalid[container]["line_cost"] = corruption

    error = "reward components" if corruption in ("missing", "extra") else "line_cost"
    with pytest.raises(ValueError, match=error):
        environment.set_state(invalid)

    np.testing.assert_equal(environment.get_state(), original)


@pytest.mark.parametrize("coefficient", [None, True, False, -0.1, np.nan, np.inf, -np.inf,
                                         "0.1", [0.1], 0.1j])
def test_invalid_line_coefficient_is_rejected(base_config, coefficient):
    base_config["reward"]["line_cost_coefficient"] = coefficient
    with pytest.raises(ValueError, match="line_cost_coefficient"):
        InterceptionEnv(base_config)


@pytest.mark.parametrize("shaping", [False, True])
def test_legacy_a_b_environment_behavior_and_snapshot_schema_stay_compatible(base_config, shaping):
    control_config = deepcopy(base_config)
    if shaping:
        control_config["reward"]["best_so_far_shaping"] = dict(length_scale_m=20.0, beta=100.0)
    control = InterceptionEnv(control_config, seed=1001)
    _, old_info = control.reset(angle_degrees=-20.0)
    expected_fields = REWARD_FIELDS + (("near_shaping",) if shaping else ())
    assert control.reward_fields == expected_fields
    assert "line_segment_distance" not in old_info
    assert "line_cost" not in control.get_state()["episode_reward_components"]
    clone = InterceptionEnv(control_config, seed=1)
    clone.set_state(control.get_state())
    for action in np.random.default_rng(412).uniform(-1, 1, (10, 3)):
        np.testing.assert_equal(control.step(action), clone.step(action))

    zero_config = deepcopy(control_config)
    zero_config["reward"]["line_cost_coefficient"] = 0.0
    zero = InterceptionEnv(zero_config, seed=1001)
    old = InterceptionEnv(control_config, seed=1001)
    obs_old, info_old = old.reset(angle_degrees=-20.0)
    obs_new, info_new = zero.reset(angle_degrees=-20.0)
    np.testing.assert_array_equal(obs_old, obs_new)
    np.testing.assert_array_equal(info_old["states"], info_new["states"])
    for action in np.random.default_rng(412).uniform(-1, 1, (10, 3)):
        first, second = old.step(action), zero.step(action)
        np.testing.assert_array_equal(first[0], second[0])
        assert first[1:4] == second[1:4]
        np.testing.assert_array_equal(first[4]["states"], second[4]["states"])
        for field in expected_fields:
            assert first[4]["reward_components"][field] == second[4]["reward_components"][field]
    np.testing.assert_equal(old.rng.bit_generator.state, zero.rng.bit_generator.state)


@pytest.fixture
def training_config():
    result = load_configuration(ROOT / "configs/ppo_reward_ab.yaml")
    result["environment_config"]["reward"]["line_cost_coefficient"] = 0.1
    # These small networks and budgets are engineering fixtures.
    result["environment_config"]["environment"]["max_episode_steps"] = 13
    result["training"].update(total_env_steps=48, num_envs=2, rollout_steps=8,
                              evaluation_interval_steps=12, validation_episodes=3)
    result["network"].update(actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8])
    result["ppo"].update(update_epochs=1, minibatch_size=8)
    return result


def evaluation_fixture(*args):
    return dict(capture_rate=0.0, defense_failure_rate=0.0, timeout_rate=1.0,
                average_interception_time=None, average_min_distance=400.0, average_return=-5.0)


def test_partial_rollout_restore_preserves_models_optimizer_rewards_and_csv(training_config, tmp_path):
    continuous, paused, restored = (tmp_path / name for name in ("continuous", "paused", "restored"))
    train(training_config, continuous, 1001, evaluation_fn=evaluation_fixture)
    summary = train(training_config, paused, 1001, stop_at_steps=20, evaluation_fn=evaluation_fixture)
    assert summary["pending_rollout_samples"] == 4
    checkpoint = paused / "checkpoints/paused_000020.pt"
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for environment in bundle["training_state"]["vector_environments"]["environments"]:
        assert environment["step_count"] == 10
        assert environment["episode_reward_components"]["line_cost"] < 0
    train(training_config, restored, 1001, resume_checkpoint=checkpoint, evaluation_fn=evaluation_fixture)
    left = torch.load(continuous / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    right = torch.load(restored / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    for checkpoint in (left, right):
        checkpoint["agent"]["config"].pop("output_dir", None)
    for key in ("agent", "training_state", "rng_state"):
        assert _state_hash(left[key]) == _state_hash(right[key])
    for path in continuous.glob("*.csv"):
        assert path.read_bytes() == (restored / path.name).read_bytes()
    rows = read_csv(continuous / "episodes.csv")
    assert rows and "line_cost_return" in rows[0]
    for row in rows:
        assert float(row["line_cost_return"]) < 0
        assert float(row["episode_return"]) == pytest.approx(float(row["original_return"])
            + float(row["near_shaping_return"]) + float(row["line_cost_return"]), abs=1e-10)


def test_evaluation_npz_and_csv_save_actual_line_geometry_and_components(training_config, tmp_path):
    agent = PPOAgent(training_config)
    output = tmp_path / "evaluation"
    evaluate_angles(agent, training_config["environment_config"], [-20.0, 20.0], 2001, output)
    for row in read_csv(output / "episodes.csv"):
        with np.load(output / row["trajectory"], allow_pickle=False) as trajectory:
            states = trajectory["states"]
            expected_distances = segment_distances(states[:, 0, :3], states[:, 1, :3],
                training_config["environment_config"]["mission"]["base_position"])
            np.testing.assert_allclose(trajectory["line_distances"], expected_distances, rtol=0, atol=1e-12)
            expected_cost = line_cost(expected_distances, trajectory["time"], 0.1)
            np.testing.assert_allclose(trajectory["line_cost"], expected_cost, rtol=0, atol=1e-12)
            assert len(trajectory["line_distances"]) == len(trajectory["rewards"]) + 1
            assert float(row["line_cost_return"]) == pytest.approx(trajectory["line_cost"].sum(), abs=1e-10)
            assert float(row["line_segment_distance"]) == pytest.approx(expected_distances[-1])
            assert float(row["episode_return"]) == pytest.approx(float(row["original_return"])
                + float(row["near_shaping_return"]) + float(row["line_cost_return"]), abs=1e-10)
            metadata = json.loads(trajectory["metadata_json"].item())
            assert metadata["line_cost_return"] == pytest.approx(trajectory["line_cost"].sum(), abs=1e-10)
