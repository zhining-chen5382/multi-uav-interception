"""Distribution, return estimation, and learning checks for the shared MAPPO agent."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import yaml

from algorithms.mappo import MAPPOAgent, RolloutBuffer, build_input_scales
from algorithms.mappo.networks import CentralizedCritic, SharedActor
from envs import PursuitEnv


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actor():
    torch.manual_seed(19)
    return SharedActor(
        observation_dim=21,
        action_dim=3,
        hidden_sizes=[16, 16],
        initial_log_std=-0.5,
        min_log_std=-5.0,
        max_log_std=2.0,
        hidden_gain=np.sqrt(2.0),
        output_gain=0.01,
    )


@pytest.fixture
def small_config():
    with (ROOT / "configs" / "mappo_baseline.yaml").open(encoding="utf-8") as stream:
        config = deepcopy(yaml.safe_load(stream))
    # The test fixture changes network/update cost only, never the environment.
    config["network"]["actor_hidden_sizes"] = [16, 16]
    config["network"]["critic_hidden_sizes"] = [16, 16]
    config["training"]["rollout_steps"] = 8
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 4
    return config


def _add_transition(buffer, reward, value, next_value, terminated=False, truncated=False):
    buffer.add(
        observations=np.zeros((3, 21), dtype=np.float32),
        state=np.zeros(24, dtype=np.float32),
        raw_actions=np.zeros((3, 3), dtype=np.float32),
        log_probs=np.zeros(3, dtype=np.float32),
        reward=reward,
        value=value,
        next_value=next_value,
        terminated=terminated,
        truncated=truncated,
    )


def _collect_rollout(agent, steps=8):
    env = PursuitEnv(ROOT / "configs" / "baseline_3a1c.yaml")
    observations, _ = env.reset(seed=17)
    buffer = RolloutBuffer(steps, 3, 21, 24, 3)
    for _ in range(steps):
        state = env.get_state()
        actions, log_probs, raw_actions, value = agent.act(observations, state)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        buffer.add(
            observations=observations,
            state=state,
            raw_actions=raw_actions,
            log_probs=log_probs,
            reward=float(rewards[0]),
            value=value,
            next_value=agent.value(env.get_state()),
            terminated=terminated,
            truncated=truncated,
        )
        observations = next_observations
        if terminated or truncated:
            observations, _ = env.reset()
    buffer.compute_returns(agent.config["ppo"]["gamma"], agent.config["ppo"]["gae_lambda"])
    return buffer, observations, env.get_state()


@pytest.mark.parametrize("shape", [(21,), (3, 21), (2, 3, 21)])
def test_actor_accepts_individual_and_batched_local_observations(actor, shape):
    observations = torch.randn(shape)
    assert actor(observations).shape == shape[:-1] + (3,)
    actions, log_probs, raw_actions = actor.sample(observations)
    assert actions.shape == raw_actions.shape == shape[:-1] + (3,)
    assert log_probs.shape == shape[:-1]
    assert torch.isfinite(actions).all()
    assert torch.isfinite(raw_actions).all()
    assert torch.isfinite(log_probs).all()
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    torch.testing.assert_close(actions, torch.tanh(raw_actions))


@pytest.mark.parametrize("shape", [(24,), (4, 24), (2, 3, 24)])
def test_critic_produces_one_centralized_value_per_state(shape):
    critic = CentralizedCritic(
        state_dim=24, hidden_sizes=[16, 16], hidden_gain=np.sqrt(2.0), output_gain=1.0
    )
    values = critic(torch.randn(shape))
    assert values.shape == shape[:-1]
    assert torch.isfinite(values).all()


def test_log_probability_matches_independent_transformed_distribution(actor):
    observations = torch.randn(4, 21)
    raw_actions = torch.tensor([
        [-0.4, 0.2, 0.6], [0.1, -0.8, 0.0],
        [0.5, 0.3, -0.2], [-0.1, 0.7, -0.5],
    ])
    distribution = actor.distribution(observations)
    reference = Independent(
        TransformedDistribution(distribution, [TanhTransform(cache_size=1)]), 1
    )
    actual, _ = actor.evaluate_actions(observations, raw_actions)
    expected = reference.log_prob(torch.tanh(raw_actions))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_saved_raw_actions_reproduce_old_log_prob_and_unit_ppo_ratio(actor):
    observations = torch.randn(8, 21)
    _, old_log_probs, raw_actions = actor.sample(observations)
    current_log_probs, _ = actor.evaluate_actions(observations, raw_actions.detach())
    ratio = torch.exp(current_log_probs - old_log_probs.detach())
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-6, atol=1e-6)


def test_entropy_estimate_uses_fresh_squashed_distribution_samples(actor):
    observations = torch.randn(8, 21)
    distribution = actor.distribution(observations)
    reference = Independent(
        TransformedDistribution(distribution, [TanhTransform(cache_size=1)]), 1
    )
    torch.manual_seed(31)
    expected = -reference.log_prob(reference.rsample())
    torch.manual_seed(31)
    _, actual = actor.evaluate_actions(observations, torch.zeros(8, 3))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_saturated_policy_can_have_negative_differential_entropy(actor, monkeypatch):
    distribution = torch.distributions.Normal(torch.full((4, 3), 20.0), torch.ones(4, 3))
    monkeypatch.setattr(actor, "distribution", lambda observations: distribution)
    _, entropy = actor.evaluate_actions(torch.zeros(4, 21), torch.full((4, 3), 20.0))
    assert torch.isfinite(entropy).all()
    assert torch.all(entropy < 0.0)


def test_saturated_actions_keep_finite_log_prob_without_inverse_tanh(actor):
    observations = torch.zeros(2, 21)
    raw_actions = torch.tensor([[30.0, -30.0, 30.0], [40.0, -40.0, 40.0]])
    # Both rows squash to identical boundary actions. Keeping the original raw
    # sample must still give different likelihoods; inverting actions loses it.
    torch.testing.assert_close(torch.tanh(raw_actions[0]), torch.tanh(raw_actions[1]))
    actual, entropy = actor.evaluate_actions(observations, raw_actions)
    distribution = actor.distribution(observations)
    magnitude = raw_actions.abs()
    log_jacobian = np.log(4.0) - 2.0 * magnitude - 2.0 * torch.log1p(torch.exp(-2.0 * magnitude))
    expected = (distribution.log_prob(raw_actions) - log_jacobian).sum(dim=-1)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(entropy).all()
    assert actual[0] != actual[1]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


def test_squashed_entropy_has_finite_gradient(actor):
    observations = torch.randn(16, 21)
    _, _, raw_actions = actor.sample(observations)
    _, entropy = actor.evaluate_actions(observations, raw_actions.detach())
    assert entropy.shape == (16,)
    assert torch.isfinite(entropy).all()
    (-entropy.mean()).backward()
    gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.any(gradient != 0) for gradient in gradients)


def test_deterministic_actor_actions_repeat_without_global_state(actor):
    observations = torch.randn(3, 21)
    first = actor.sample(observations, deterministic=True)
    second = actor.sample(observations, deterministic=True)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[0], torch.tanh(actor(observations)))


def test_input_scaling_is_opt_in_and_respects_observation_order(baseline_config, small_config):
    assert small_config["normalization"]["observation"] is False
    assert small_config["normalization"]["global_state"] is False
    observation_scale, state_scale = build_input_scales(
        baseline_config, small_config["normalization"]
    )
    np.testing.assert_array_equal(observation_scale, np.ones(21))
    np.testing.assert_array_equal(state_scale, np.ones(24))
    observation_scale, state_scale = build_input_scales(
        baseline_config, {"observation": True, "global_state": True}
    )
    assert observation_scale.shape == (21,)
    assert state_scale.shape == (24,)
    assert observation_scale.dtype == state_scale.dtype == np.float32
    assert np.isfinite(observation_scale).all() and np.all(observation_scale > 0)
    assert np.isfinite(state_scale).all() and np.all(state_scale > 0)
    np.testing.assert_allclose(observation_scale, [
        20, 20, 8, 1500, 1500, 1500, 40, 40, 16,
        1500, 1500, 1500, 1500, 1500, 1500, 40, 40, 16, 40, 40, 16,
    ])
    np.testing.assert_allclose(state_scale, [750, 750, 750, 20, 20, 8] * 4)


def test_rollout_buffer_shapes_and_reset():
    buffer = RolloutBuffer(8, 3, 21, 24, 3)
    expected_shapes = {
        "observations": (8, 3, 21), "states": (8, 24),
        "raw_actions": (8, 3, 3), "log_probs": (8, 3),
        "rewards": (8,), "values": (8,), "next_values": (8,),
        "terminated": (8,), "truncated": (8,),
        "advantages": (8,), "returns": (8,),
    }
    for name, shape in expected_shapes.items():
        assert getattr(buffer, name).shape == shape
    _add_transition(buffer, 1.0, 0.5, 0.6)
    assert buffer.size == 1
    buffer.reset()
    assert buffer.size == 0


def test_capture_transition_does_not_bootstrap_from_terminal_state():
    buffer = RolloutBuffer(1, 3, 21, 24, 3)
    _add_transition(buffer, reward=2.0, value=0.5, next_value=999.0, terminated=True)
    advantages, returns = buffer.compute_returns(gamma=0.9, gae_lambda=0.8)
    np.testing.assert_allclose(advantages, [1.5])
    np.testing.assert_allclose(returns, [2.0])


def test_timeout_bootstraps_final_state_and_breaks_gae_episode_chain():
    buffer = RolloutBuffer(3, 3, 21, 24, 3)
    _add_transition(buffer, reward=1.0, value=0.5, next_value=1.0)
    _add_transition(buffer, reward=2.0, value=1.0, next_value=10.0, truncated=True)
    # This is a new episode: its advantage must not enter the timeout above.
    _add_transition(buffer, reward=3.0, value=3.0, next_value=4.0)
    advantages, returns = buffer.compute_returns(gamma=0.9, gae_lambda=0.8)
    np.testing.assert_allclose(advantages, [8.6, 10.0, 3.6], rtol=1e-6)
    np.testing.assert_allclose(returns, [9.1, 11.0, 6.6], rtol=1e-6)
    assert np.isfinite(advantages).all()
    assert np.isfinite(returns).all()


def test_unfinished_rollout_bootstraps_next_value():
    buffer = RolloutBuffer(2, 3, 21, 24, 3)
    _add_transition(buffer, reward=1.0, value=0.5, next_value=1.0)
    _add_transition(buffer, reward=2.0, value=1.0, next_value=4.0)
    advantages, returns = buffer.compute_returns(gamma=0.9, gae_lambda=0.8)
    np.testing.assert_allclose(advantages, [4.712, 4.6], rtol=1e-6)
    np.testing.assert_allclose(returns, [5.212, 5.6], rtol=1e-6)


def test_ppo_update_changes_actor_and_critic_with_finite_metrics(small_config):
    torch.manual_seed(23)
    agent = MAPPOAgent(small_config, device="cpu")
    buffer, observations, _ = _collect_rollout(agent)
    before_actor = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    before_critic = [parameter.detach().clone() for parameter in agent.critic.parameters()]
    metrics = agent.update(buffer)
    required = {
        "actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
        "actor_grad_norm", "critic_grad_norm",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(metrics[key]) for key in required)
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert any(not torch.equal(old, new) for old, new in zip(before_actor, agent.actor.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(before_critic, agent.critic.parameters()))
    actions, log_probs, raw_actions, value = agent.act(observations)
    assert actions.shape == raw_actions.shape == (3, 3)
    assert log_probs.shape == (3,)
    assert value is None
    assert np.isfinite(actions).all()
    assert np.all(np.abs(actions) <= 1.0)


def test_checkpoint_restores_deterministic_actions_values_and_optimizer(
    small_config, baseline_config, tmp_path
):
    torch.manual_seed(29)
    small_config["normalization"]["observation"] = True
    small_config["normalization"]["global_state"] = True
    observation_scale, state_scale = build_input_scales(
        baseline_config, small_config["normalization"]
    )
    agent = MAPPOAgent(
        small_config, observation_scale=observation_scale,
        state_scale=state_scale, device="cpu",
    )
    buffer, observations, state = _collect_rollout(agent)
    agent.update(buffer)
    before = agent.act(observations, state, deterministic=True)
    checkpoint = tmp_path / "checkpoint.pt"
    agent.save(checkpoint, training_state={"total_steps": 8}, metadata={"seed": 29})
    restored, payload = MAPPOAgent.from_checkpoint(checkpoint, device="cpu")
    after = restored.act(observations, state, deterministic=True)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert before[3] == after[3]
    assert payload["training_state"]["total_steps"] == 8
    assert payload["metadata"]["seed"] == 29
    assert restored.actor_optimizer.state_dict()["state"]
    assert restored.critic_optimizer.state_dict()["state"]
    torch.testing.assert_close(agent.actor.input_scale, restored.actor.input_scale)
    torch.testing.assert_close(agent.critic.input_scale, restored.critic.input_scale)
