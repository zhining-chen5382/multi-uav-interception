"""Isolation and numerical checks for the fixed input-scaling experiment."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.mappo import MAPPOAgent, RolloutBuffer, build_input_scales
from envs import PursuitEnv


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs" / "baseline_3a1c.yaml"


@pytest.fixture
def normalized_agent_config():
    with (ROOT / "configs" / "mappo_baseline.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    # Small networks and one short update keep these behavioral checks cheap.
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 4
    config["normalization"]["observation"] = True
    config["normalization"]["global_state"] = True
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield config
    torch.set_num_threads(original_threads)


def _agent(config, environment_config):
    observation_scale, state_scale = build_input_scales(
        environment_config, config["normalization"]
    )
    return MAPPOAgent(
        config, observation_scale=observation_scale, state_scale=state_scale, device="cpu"
    )


def test_scaling_leaves_raw_environment_observations_and_state_unchanged(
    normalized_agent_config, baseline_config
):
    agent = _agent(normalized_agent_config, baseline_config)
    env = PursuitEnv(ENV_CONFIG)
    reference_env = PursuitEnv(ENV_CONFIG)
    observations, _ = env.reset(seed=0)
    reference_observations, _ = reference_env.reset(seed=0)
    state = env.get_state()
    saved_observations, saved_state = observations.copy(), state.copy()
    saved_configuration = deepcopy(baseline_config)

    agent.act(observations, state, deterministic=True)

    assert observations.shape == (3, 21)
    assert state.shape == (24,)
    # Initial target separation is still expressed in metres, not unit scales.
    assert np.max(np.abs(observations)) > 100
    np.testing.assert_array_equal(observations, saved_observations)
    np.testing.assert_array_equal(state, saved_state)
    np.testing.assert_array_equal(env.get_state(), saved_state)
    assert baseline_config == saved_configuration

    # With identical physical commands, calling a scaled policy must have no
    # side effect on the environment's subsequent transitions or rewards.
    commands = np.random.default_rng(13).uniform(-1, 1, size=(3, 3))
    actual = env.step(commands)
    expected = reference_env.step(commands)
    np.testing.assert_array_equal(reference_observations, saved_observations)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:4] == expected[2:4]
    np.testing.assert_array_equal(env.get_state(), reference_env.get_state())


def test_same_seed_changes_only_scale_buffers_not_initial_network_parameters(
    normalized_agent_config, baseline_config
):
    unscaled_config = deepcopy(normalized_agent_config)
    unscaled_config["normalization"].update(observation=False, global_state=False)
    torch.manual_seed(0)
    unscaled = _agent(unscaled_config, baseline_config)
    torch.manual_seed(0)
    scaled = _agent(normalized_agent_config, baseline_config)

    for original, normalized in ((unscaled.actor, scaled.actor),
                                 (unscaled.critic, scaled.critic)):
        original_parameters = dict(original.named_parameters())
        normalized_parameters = dict(normalized.named_parameters())
        assert original_parameters.keys() == normalized_parameters.keys()
        assert original_parameters
        for name, parameter in original_parameters.items():
            assert torch.equal(parameter, normalized_parameters[name]), name
        differing_entries = {
            name for name, value in original.state_dict().items()
            if not torch.equal(value, normalized.state_dict()[name])
        }
        assert differing_entries == {"input_scale"}


def test_checkpoint_preserves_enabled_config_and_nonunit_input_scales(
    normalized_agent_config, baseline_config, tmp_path
):
    torch.manual_seed(0)
    agent = _agent(normalized_agent_config, baseline_config)
    env = PursuitEnv(ENV_CONFIG)
    observations, _ = env.reset(seed=0)
    state = env.get_state()
    expected = agent.act(observations, state, deterministic=True)
    checkpoint = tmp_path / "normalized.pt"
    agent.save(checkpoint)
    restored, payload = MAPPOAgent.from_checkpoint(checkpoint, device="cpu")

    for configuration in (payload["config"], restored.config):
        assert configuration["normalization"]["observation"] is True
        assert configuration["normalization"]["global_state"] is True
        assert configuration["normalization"]["reward_scale"] == 1.0
    assert torch.any(restored.actor.input_scale != 1)
    assert torch.any(restored.critic.input_scale != 1)
    assert torch.equal(agent.actor.input_scale, restored.actor.input_scale)
    assert torch.equal(agent.critic.input_scale, restored.critic.input_scale)
    actual = restored.act(observations, state, deterministic=True)
    for original_array, restored_array in zip(expected[:3], actual[:3]):
        np.testing.assert_array_equal(original_array, restored_array)
    assert expected[3] == actual[3]


def test_scaled_policy_short_physical_rollout_and_update_remain_finite(
    normalized_agent_config, baseline_config
):
    torch.manual_seed(0)
    agent = _agent(normalized_agent_config, baseline_config)
    env = PursuitEnv(ENV_CONFIG)
    observations, _ = env.reset(seed=0)
    buffer = RolloutBuffer(8, 3, 21, 24, 3)
    actor_scale = agent.actor.input_scale.clone()
    critic_scale = agent.critic.input_scale.clone()
    before_actor = [value.detach().clone() for value in agent.actor.parameters()]
    before_critic = [value.detach().clone() for value in agent.critic.parameters()]

    for _ in range(buffer.capacity):
        state = env.get_state()
        actions, log_probs, raw_actions, value = agent.act(observations, state)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        buffer.add(
            observations, state, raw_actions, log_probs, float(rewards[0]), value,
            agent.value(env.get_state()), terminated, truncated,
        )
        observations = next_observations
        assert np.isfinite(actions).all() and np.all(np.abs(actions) <= 1)
        if terminated or truncated:
            observations, _ = env.reset()

    # The buffer intentionally mixes hundreds of metres with small velocities.
    # Scaling belongs to the networks; storing already scaled inputs here would
    # apply the conversion twice and invalidate train/evaluation consistency.
    assert np.max(np.abs(buffer.observations)) > 100
    assert np.any((np.abs(buffer.observations) > 0) &
                  (np.abs(buffer.observations) < 1))
    buffer.compute_returns(agent.ppo["gamma"], agent.ppo["gae_lambda"])
    metrics = agent.update(buffer)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before, after) for before, after in
               zip(before_actor, agent.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in
               zip(before_critic, agent.critic.parameters()))
    for network in (agent.actor, agent.critic):
        assert all(torch.isfinite(value).all() for value in network.parameters())
        assert all(value.grad is None or torch.isfinite(value.grad).all()
                   for value in network.parameters())
    assert torch.equal(actor_scale, agent.actor.input_scale)
    assert torch.equal(critic_scale, agent.critic.input_scale)
    final = agent.act(observations, env.get_state(), deterministic=True)
    assert all(np.isfinite(value).all() for value in final[:3])
    assert np.isfinite(final[3])
