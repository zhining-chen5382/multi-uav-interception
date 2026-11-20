"""User-confirmed production normalization and applied-acceleration contract.

These checks read the actual formal YAML. Synthetic physical states exercise
every field, while the real integrator and evaluation entry points verify the
same contract during execution. They do not train or evaluate a learned model.
"""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
import yaml

from algorithms.ppo import PPOAgent
from envs import interception_env
from envs.dynamics import integrate, scale_action
from envs.interception_env import InterceptionEnv, OBSERVATION_FIELDS
from scripts.run_formal_experiment import run_experiment
from utils.formal_ppo_config import load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import evaluate_angles
from utils.formal_ppo_training import IndependentEnvironments, _record_violations, train, validate_config


CONFIRMED_SCALES = np.array([
    20., 20., 8., 3., 3., 2., 1500., 1500., 1500.,
    40., 40., 14., 500., 500., 500., 1.,
])
CONFIRMED_FIELDS = (
    "own_vx", "own_vy", "own_vz", "own_ax", "own_ay", "own_az",
    "relative_x", "relative_y", "relative_z", "relative_vx", "relative_vy", "relative_vz",
    "attacker_base_x", "attacker_base_y", "attacker_base_z", "remaining_time_fraction",
)
SIGNED_RAW_OBSERVATION = np.array([
    4., -6., 2., 1.2, -.9, .8, 360., -80., 40.,
    -12., 16., -5., 460., -200., 70., .75,
])


@pytest.fixture
def production_config():
    return load_formal_config()


@pytest.fixture
def signed_state(production_config):
    environment = InterceptionEnv(production_config["environment_config"], seed=1001)
    environment.reset(angle_degrees=0.)
    environment.states[:] = [[100., -120., 30., 4., -6., 2.],
                             [460., -200., 70., -8., 10., -3.]]
    environment.accelerations[0] = [1.2, -.9, .8]
    # Deliberately different diagnostics catch an accidental dv/dt substitution.
    environment.realized_accelerations[0] = [-.4, .2, -.6]
    environment.step_count = 125
    return environment


def test_actual_production_configuration_records_exact_user_confirmation(production_config):
    normalization = production_config["environment_config"]["normalization"]
    assert normalization["confirmed"] is True
    assert normalization["acceleration_kind"] == "applied"
    assert normalization["confirmation_source"]
    np.testing.assert_array_equal(normalization["scales"], CONFIRMED_SCALES)
    assert OBSERVATION_FIELDS == CONFIRMED_FIELDS
    assert validate_readiness(production_config) is production_config
    assert validate_config(production_config) is production_config


@pytest.mark.parametrize("index", range(16), ids=CONFIRMED_FIELDS)
def test_every_observation_field_uses_its_exact_confirmed_divisor(signed_state, index):
    raw = signed_state.raw_observation()
    normalized = signed_state._observe()
    assert raw.shape == normalized.shape == (16,)
    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(raw, SIGNED_RAW_OBSERVATION)
    assert raw[index] != 0
    assert signed_state.scales[index] == CONFIRMED_SCALES[index]
    assert normalized[index] == np.float32(SIGNED_RAW_OBSERVATION[index] / CONFIRMED_SCALES[index])


def test_normalization_preserves_zero_and_ignores_observation_history(signed_state, production_config):
    reference = signed_state._observe().copy()
    # Different observation history must affect diagnostics alone.
    signed_state.states[1, :3] = [-600., 600., -600.]
    for _ in range(10):
        signed_state._observe()
    signed_state.states[1, :3] = [460., -200., 70.]
    np.testing.assert_array_equal(signed_state._observe(), reference)
    fresh = InterceptionEnv(production_config["environment_config"], seed=5)
    initial, _ = fresh.reset(angle_degrees=0.)
    np.testing.assert_array_equal(initial[:6], np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(initial[9:12], np.zeros(3, dtype=np.float32))
    assert initial[15] == 1.


@pytest.mark.parametrize("axis,cap,sign", [(0, 20., 1.), (1, 20., 1.), (2, 8., 1.), (2, -6., -1.)],
                         ids=["vx-cap", "vy-cap", "climb-cap", "descent-cap"])
def test_speed_saturation_keeps_applied_observation_and_next_jerk_source(
        production_config, monkeypatch, axis, cap, sign):
    environment = InterceptionEnv(production_config["environment_config"], seed=1001)
    environment.reset(angle_degrees=0.)
    environment.states[0, 3 + axis] = cap
    action = np.zeros(3)
    action[axis] = sign
    first, _, done, _, info = environment.step(action)
    assert not done
    first_applied = np.zeros(3)
    first_applied[axis] = sign * .5
    np.testing.assert_array_equal(environment.accelerations[0], first_applied)
    np.testing.assert_array_equal(info["raw_observation"][3:6], first_applied)
    np.testing.assert_array_equal(info["realized_accelerations"][0], np.zeros(3))
    np.testing.assert_array_equal(first[3:6], (first_applied / CONFIRMED_SCALES[3:6]).astype(np.float32))
    stored_accelerations = environment.accelerations
    recorded_previous = []

    def inspect_integrate(state, command, previous_acceleration, dt, limits):
        if limits is environment.pursuer_limits:
            assert np.shares_memory(previous_acceleration, stored_accelerations)
            recorded_previous.append(previous_acceleration.copy())
        return integrate(state, command, previous_acceleration, dt, limits)

    monkeypatch.setattr(interception_env, "integrate", inspect_integrate)
    before = environment.states[0].copy()
    command = scale_action(action, environment.pursuer_limits)
    expected_state, expected_applied = integrate(before, command, first_applied,
                                                environment.dt, environment.pursuer_limits)
    _, wrong_applied = integrate(before, command, np.zeros(3), environment.dt, environment.pursuer_limits)
    second, _, done, _, second_info = environment.step(action)
    assert not done and len(recorded_previous) == 1
    np.testing.assert_array_equal(recorded_previous[0], first_applied)
    np.testing.assert_array_equal(environment.states[0], expected_state)
    np.testing.assert_array_equal(environment.accelerations[0], expected_applied)
    np.testing.assert_array_equal(second_info["raw_observation"][3:6], expected_applied)
    np.testing.assert_array_equal(second[3:6], (expected_applied / CONFIRMED_SCALES[3:6]).astype(np.float32))
    assert expected_applied[axis] == sign
    assert not np.array_equal(expected_applied, wrong_applied)


def test_over_scale_values_remain_visible_and_reach_training_diagnostics(production_config):
    environment = InterceptionEnv(production_config["environment_config"], seed=1001)
    environment.reset(angle_degrees=0.)
    environment.states[:, :3] = [[-700., 0., -700.], [700., 0., 700.]]
    observation = environment._observe()
    info = environment._info()
    np.testing.assert_array_equal(observation[[12, 14]], np.array([1.4, 1.4], dtype=np.float32))
    assert (observation[[12, 14]] > 1).all()
    np.testing.assert_array_equal(np.flatnonzero(info["normalization_violation_mask"]), [12, 14])
    event = info["normalization_violations"]
    assert event["fields"] == ["attacker_base_x", "attacker_base_z"]
    assert event["raw_values"] == [700., 700.]
    assert event["normalized_values"] == [1.4, 1.4]
    rows = []
    _record_violations(rows, info, env_steps=1000, env_index=3)
    assert [row["field_index"] for row in rows] == [12, 14]
    assert [row["raw_value"] for row in rows] == [700., 700.]
    assert [row["normalized_value"] for row in rows] == [1.4, 1.4]
    assert all(row["env_steps"] == 1000 and row["env_index"] == 3 for row in rows)


def test_actor_and_critic_consume_already_normalized_input_once(production_config):
    with torch.random.fork_rng():
        agent = PPOAgent(production_config)
        observation = torch.tensor(SIGNED_RAW_OBSERVATION / CONFIRMED_SCALES, dtype=torch.float32).unsqueeze(0)
        np.testing.assert_array_equal(agent.actor.input_scale.cpu().numpy(), np.ones(16))
        np.testing.assert_array_equal(agent.critic.input_scale.cpu().numpy(), np.ones(16))
        with torch.no_grad():
            torch.testing.assert_close(agent.actor(observation), agent.actor.mean(observation), rtol=0, atol=0)
            torch.testing.assert_close(agent.critic(observation), agent.critic.value(observation).squeeze(-1), rtol=0, atol=0)


class ZeroControlAgent:
    """Deterministic test control exercises evaluation wiring without training."""

    def __init__(self):
        self.actor = torch.nn.Linear(16, 3)
        self.critic = torch.nn.Linear(16, 1)

    def act(self, observations, deterministic=False):
        assert deterministic is True
        return dict(actions=np.zeros((len(observations), 3), dtype=np.float32))


@pytest.mark.parametrize("role", ["validation", "test"])
def test_training_and_both_evaluation_routes_use_one_production_contract(production_config, tmp_path, role):
    environment_config = production_config["environment_config"]
    vector = IndependentEnvironments(environment_config, production_config["training"]["num_envs"], 1001)
    observations, infos = vector.reset()
    for environment, observation, info in zip(vector.envs, observations, infos):
        assert environment.config == environment_config
        np.testing.assert_array_equal(environment.scales, CONFIRMED_SCALES)
        np.testing.assert_array_equal(observation, (info["raw_observation"] / CONFIRMED_SCALES).astype(np.float32))
    output = tmp_path / role
    seed = production_config["experiment"][role + "_seed"]
    with torch.random.fork_rng():
        agent = ZeroControlAgent()
        summary = evaluate_angles(agent, environment_config, [-23.5, 17.25], seed, output)
    assert summary["normalization"] == environment_config["normalization"]
    saved = yaml.safe_load((output / "environment_config.yaml").read_text(encoding="utf-8"))
    assert saved == environment_config
    for path in (output / "trajectories").glob("*.npz"):
        with np.load(path, allow_pickle=False) as trajectory:
            raw = trajectory["raw_observations"]
            np.testing.assert_array_equal(trajectory["observations"], (raw / CONFIRMED_SCALES).astype(np.float32))
            np.testing.assert_array_equal(raw[1:, 3:6], trajectory["applied_accelerations"][:, 0])
            assert json.loads(trajectory["metadata_json"].item())["episode_length"] + 1 == len(raw)


@pytest.mark.parametrize("entry", ["readiness", "training-validation", "formal-suite", "single-run"])
def test_formal_entry_points_reject_realized_acceleration_before_output(production_config, tmp_path, entry):
    config = deepcopy(production_config)
    config["environment_config"]["normalization"]["acceleration_kind"] = "realized"
    output = tmp_path / "must_not_be_created"
    calls = {
        "readiness": lambda: validate_readiness(config),
        "training-validation": lambda: validate_config(config),
        "formal-suite": lambda: run_experiment(config, output),
        "single-run": lambda: train(config, output, 1001),
    }
    with pytest.raises(ValueError, match="applied"):
        calls[entry]()
    assert not output.exists()


def test_production_snapshot_restores_same_applied_observation_and_next_dynamics(production_config):
    source = InterceptionEnv(production_config["environment_config"], seed=1001)
    source.reset(angle_degrees=0.)
    source.states[0, 3:] = [20., 0., 0.]
    source.step([1., 0., 0.])
    restored = InterceptionEnv(production_config["environment_config"], seed=1002)
    restored.set_state(source.get_state())
    np.testing.assert_array_equal(restored.raw_observation(), source.raw_observation())
    source_result = source.step([1., 0., 0.])
    restored_result = restored.step([1., 0., 0.])
    np.testing.assert_array_equal(source_result[0], restored_result[0])
    np.testing.assert_array_equal(restored.accelerations, source.accelerations)
    np.testing.assert_array_equal(restored.states, source.states)
