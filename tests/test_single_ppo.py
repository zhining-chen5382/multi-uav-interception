"""Numerical and actual-update checks for the independent single-agent PPO."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch
from torch.distributions import Normal, kl_divergence

from algorithms.ppo import PPOAgent, VectorRolloutBuffer, diagonal_gaussian_kl


@pytest.fixture(autouse=True)
def isolated_torch():
    threads = torch.get_num_threads()
    rng = torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


@pytest.fixture
def config():
    return dict(device="cpu", training=dict(total_env_steps=128), network=dict(
        actor_hidden_sizes=[16, 16], critic_hidden_sizes=[16, 16], initial_std=.5,
        min_log_std=-5., max_log_std=0., hidden_gain=math.sqrt(2), actor_output_gain=.01,
        critic_output_gain=1.), ppo=dict(actor_learning_rate=3e-4,
        actor_final_learning_rate=3e-5, critic_learning_rate=1e-3,
        critic_final_learning_rate=1e-4, adam_betas=[.9, .999], adam_epsilon=1e-5,
        weight_decay=0., amsgrad=False, gamma=.99, gae_lambda=.95, clip_ratio=.2,
        update_epochs=4, minibatch_size=8, entropy_coefficient=.01,
        value_loss_coefficient=.5, max_grad_norm=.5, advantage_normalization=True,
        normalization_epsilon=1e-8, target_kl=.015, value_clipping=False,
        value_normalization=False))


def equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            equal(actual[key], value)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            equal(left, right)
    else:
        assert actual == expected


def append_transition(buffer, agent, observations, next_observations, terminal=None):
    prediction = agent.act(observations)
    reward = .4 * prediction["actions"][:, 0] - .6 * np.square(prediction["actions"]).sum(-1)
    terminals = np.zeros(buffer.num_envs, dtype=bool) if terminal is None else terminal
    buffer.add(observations, prediction["raw_actions"], prediction["log_probs"],
               prediction["means"], prediction["log_stds"], reward, prediction["values"],
               agent.value(next_observations), terminals)


def rollout(agent, steps=4, environments=8):
    buffer = VectorRolloutBuffer(steps, environments)
    observations = np.random.default_rng(101).normal(size=(steps + 1, environments, 16)).astype(np.float32)
    for index in range(steps):
        append_transition(buffer, agent, observations[index], observations[index + 1])
    buffer.compute_returns(.99, .95)
    return buffer


def test_agent_dimensions_independence_and_initial_sampling(config):
    config["network"]["actor_hidden_sizes"] = [128, 128]
    config["network"]["critic_hidden_sizes"] = [128, 128]
    agent = PPOAgent(config)
    layers = [layer for layer in agent.actor.mean if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(16, 128), (128, 128), (128, 3)]
    critic_layers = [layer for layer in agent.critic.value if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in critic_layers] == [(16, 128), (128, 128), (128, 1)]
    assert len([layer for layer in agent.actor.mean if isinstance(layer, torch.nn.Tanh)]) == 2
    assert set(map(id, agent.actor.parameters())).isdisjoint(map(id, agent.critic.parameters()))
    assert agent.actor.log_std.shape == (3,) and agent.actor.log_std.requires_grad
    torch.testing.assert_close(agent.actor.log_std.exp(), torch.full((3,), .5))
    prediction = agent.act(np.zeros((8, 16), dtype=np.float32))
    assert prediction["actions"].shape == prediction["means"].shape == (8, 3)
    assert prediction["values"].shape == (8,)
    assert np.isfinite(prediction["actions"]).all() and np.abs(prediction["actions"]).max() <= 1
    assert not np.array_equal(prediction["actions"][0], prediction["actions"][1])
    deterministic = agent.act(np.zeros((8, 16), dtype=np.float32), deterministic=True)
    np.testing.assert_allclose(deterministic["actions"], np.tanh(deterministic["means"]), atol=1e-7)


def test_tanh_log_probability_matches_density_change_of_variables(config):
    agent = PPOAgent(config)
    mean = torch.tensor([[.2, -.1, .4]], dtype=torch.float64)
    std = torch.tensor([[.5, .3, .8]], dtype=torch.float64)
    raw = torch.tensor([[-1.1, .2, 1.4]], dtype=torch.float64)
    normal_log = -.5 * ((raw - mean) / std).square() - std.log() - .5 * math.log(2 * math.pi)
    expected = (normal_log - torch.log1p(-raw.tanh().square())).sum(-1)
    actual = agent.actor.log_prob_from_raw(Normal(mean, std), raw)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    extreme = torch.tensor([[-100., 0., 100.]])
    assert torch.isfinite(agent.actor.log_prob_from_raw(Normal(torch.zeros(3), torch.ones(3)), extreme)).all()


def test_initial_ratio_is_one_and_stored_gaussians_are_snapshots(config):
    agent = PPOAgent(config)
    observations = np.random.default_rng(55).normal(size=(8, 16)).astype(np.float32)
    prediction = agent.act(observations)
    distribution = agent.actor.distribution(agent._tensor(observations))
    current_log = agent.actor.log_prob_from_raw(distribution, agent._tensor(prediction["raw_actions"]))
    np.testing.assert_array_equal(current_log.detach().numpy(), prediction["log_probs"])
    torch.testing.assert_close((current_log - agent._tensor(prediction["log_probs"])).exp(), torch.ones(8))
    old_logs = prediction["log_stds"].copy()
    with torch.no_grad():
        agent.actor.log_std.add_(.1)
    np.testing.assert_array_equal(prediction["log_stds"], old_logs)


def test_mc_squashed_entropy_matches_gaussian_quadrature(config):
    agent = PPOAgent(config)
    observations = torch.zeros((40000, 16))
    with torch.no_grad():
        _, entropy = agent.actor.evaluate_actions(observations, torch.zeros((40000, 3)))
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    raw = math.sqrt(2) * .5 * nodes
    expected_one_dimension = .5 * math.log(2 * math.pi * math.e * .5 ** 2)
    expected_one_dimension += float(np.dot(weights, np.log1p(-np.tanh(raw) ** 2)) / math.sqrt(math.pi))
    assert entropy.shape == (40000,)
    assert abs(float(entropy.mean()) - 3 * expected_one_dimension) < .02


@pytest.mark.parametrize("seed", [0, 13, 200])
def test_analytic_kl_matches_torch_gaussians_in_double_precision(seed):
    generator = torch.Generator().manual_seed(seed)
    old_mean = torch.randn((19, 3), generator=generator, dtype=torch.float64)
    new_mean = torch.randn((19, 3), generator=generator, dtype=torch.float64)
    old_logs = -torch.rand((19, 3), generator=generator, dtype=torch.float64) * 3
    new_logs = -torch.rand((19, 3), generator=generator, dtype=torch.float64) * 3
    expected = kl_divergence(Normal(old_mean, old_logs.exp()), Normal(new_mean, new_logs.exp())).sum(-1)
    torch.testing.assert_close(diagonal_gaussian_kl(old_mean, old_logs, new_mean, new_logs), expected,
                               rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(diagonal_gaussian_kl(old_mean, old_logs, old_mean, old_logs)) == 0


def test_eight_environment_gae_terminates_each_chain_and_bootstraps_rollout_cuts():
    buffer = VectorRolloutBuffer(2, 8)
    zeros = np.zeros((8, 3), dtype=np.float32)
    obs = np.zeros((8, 16), dtype=np.float32)
    values = np.zeros(8, dtype=np.float32)
    first_terminal = np.zeros(8, dtype=bool)
    first_terminal[0] = True
    second_terminal = np.zeros(8, dtype=bool)
    second_terminal[2] = True
    buffer.add(obs, zeros, values, zeros, zeros, np.array([1, 10, 100, 0, 0, 0, 0, 0]),
               values, values, first_terminal)
    buffer.add(obs, zeros, values, zeros, zeros, np.array([2, 20, 200, 0, 0, 0, 0, 0]),
               values, np.array([3, 3, 999, 0, 0, 0, 0, 0]), second_terminal)
    advantages, returns = buffer.compute_returns(1., 1.)
    np.testing.assert_array_equal(returns[0], [1, 33, 300, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(returns[1], [5, 23, 200, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(advantages, returns)
    assert buffer.flattened()["observations"].shape == (16, 16)


def test_partial_buffer_restore_preserves_every_old_distribution_and_counter(config):
    agent = PPOAgent(config)
    full = rollout(agent)
    state = full.state_dict()
    state["size"], state["returns_ready"] = 2, False
    restored = VectorRolloutBuffer(4, 8)
    restored.load_state_dict(state)
    equal(restored.state_dict(), state)
    state["arrays"]["means"][:] = 999
    assert not np.any(restored.means == 999)
    with pytest.raises(RuntimeError):
        restored.flattened()
    restored.compute_returns(.99, .95)
    assert restored.flattened()["observations"].shape == (16, 16)


def test_update_changes_mean_critic_and_learnable_std_and_records_every_batch(config):
    config["ppo"]["target_kl"] = 100.
    agent = PPOAgent(config)
    initial = agent.state_dict()
    buffer = rollout(agent)
    result = agent.update(buffer, env_steps=32)
    assert not torch.equal(agent.actor.log_std, initial["actor"]["log_std"])
    assert agent.actor.log_std.grad is not None
    assert any(not torch.equal(value, initial["actor"][name]) for name, value in agent.actor.state_dict().items() if name.startswith("mean."))
    assert any(not torch.equal(value, initial["critic"][name]) for name, value in agent.critic.state_dict().items())
    assert len(result["minibatches"]) == 16 and len(result["epochs"]) == 4
    assert result["early_stopped"] is False
    assert all(row["samples"] == 8 for row in result["minibatches"])
    assert all(row["minibatches"] == 4 for row in result["epochs"])
    assert all(row["analytic_kl"] == row["approx_kl"] for row in result["minibatches"])
    json.dumps(result, allow_nan=False)
    for optimizer in (agent.actor_optimizer, agent.critic_optimizer):
        assert optimizer.param_groups[0]["betas"] == (.9, .999)
        assert optimizer.param_groups[0]["eps"] == 1e-5
        assert optimizer.param_groups[0]["weight_decay"] == 0
        assert optimizer.param_groups[0]["amsgrad"] is False


def test_kl_is_after_update_and_stops_all_remaining_minibatches_and_epochs(config, monkeypatch):
    agent = PPOAgent(config)
    buffer = rollout(agent)
    steps = dict(actor=0, critic=0)
    actor_step, critic_step = agent.actor_optimizer.step, agent.critic_optimizer.step

    def moved_actor(*args, **kwargs):
        actor_step(*args, **kwargs)
        with torch.no_grad():
            agent.actor.mean[-1].bias.add_(.5)
        steps["actor"] += 1

    def counted_critic(*args, **kwargs):
        steps["critic"] += 1
        return critic_step(*args, **kwargs)

    monkeypatch.setattr(agent.actor_optimizer, "step", moved_actor)
    monkeypatch.setattr(agent.critic_optimizer, "step", counted_critic)
    result = agent.update(buffer, env_steps=32)
    assert result["early_stopped"] and result["early_stop_kl"] > .015
    assert steps == dict(actor=1, critic=1)
    assert len(result["minibatches"]) == len(result["epochs"]) == 1
    assert result["epochs"][0]["early_stopped"] is True
    assert result["minibatches"][0]["clip_fraction"] > .5


def test_fixed_std_bounds_apply_to_parameter_itself_after_ppo(config):
    agent = PPOAgent(config)
    with torch.no_grad():
        agent.actor.log_std.copy_(torch.tensor([.1, -5.1, -.7]))
    result = agent.update(rollout(agent), env_steps=32)
    assert bool((agent.actor.log_std >= -5).all() and (agent.actor.log_std <= 0).all())
    assert agent.actor.log_std.requires_grad and agent.actor.log_std.grad is not None
    json.dumps(result, allow_nan=False)


def test_kl_threshold_uses_strict_greater_than(config):
    config["ppo"]["target_kl"] = 100.
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["minibatch_size"] = 32
    reference = PPOAgent(config)
    buffer = rollout(reference)
    initial, random_state = reference.state_dict(), torch.get_rng_state().clone()
    first = reference.update(buffer, env_steps=32)
    threshold = first["last_kl"]
    assert threshold > 0
    for target, stopped in ((threshold, False), (threshold / 2, True)):
        agent = PPOAgent(config)
        agent.load_state_dict(initial)
        agent.ppo["target_kl"] = target
        torch.set_rng_state(random_state)
        result = agent.update(buffer, env_steps=32)
        assert result["last_kl"] == threshold
        assert result["early_stopped"] is stopped


def test_lr_is_environment_based_and_shared_by_actor_mean_and_std(config):
    agent = PPOAgent(config)
    assert any(parameter is agent.actor.log_std for parameter in agent.actor_optimizer.param_groups[0]["params"])
    rates = agent.scheduler.step(64)
    assert rates == pytest.approx((1.65e-4, 5.5e-4))
    agent.scheduler.step(128)
    assert agent.actor_optimizer.param_groups[0]["lr"] == 3e-5
    assert agent.critic_optimizer.param_groups[0]["lr"] == 1e-4
    with pytest.raises(ValueError):
        agent.scheduler.step(127)
    with pytest.raises(ValueError):
        agent.scheduler.step(129)


def test_agent_and_partial_rollout_restore_reproduce_next_actual_ppo_update(config):
    config["ppo"]["target_kl"] = 100.
    agent = PPOAgent(config)
    agent.update(rollout(agent, environments=2), env_steps=8)
    observations = np.random.default_rng(31).normal(size=(5, 2, 16)).astype(np.float32)
    partial = VectorRolloutBuffer(4, 2)
    for index in range(2):
        append_transition(partial, agent, observations[index], observations[index + 1])
    saved_agent, saved_buffer, saved_rng = agent.state_dict(), partial.state_dict(), torch.get_rng_state().clone()
    assert saved_agent["actor_optimizer"]["state"] and saved_agent["critic_optimizer"]["state"]
    for index in range(2, 4):
        append_transition(partial, agent, observations[index], observations[index + 1])
    partial.compute_returns(.99, .95)
    uninterrupted_metrics = agent.update(partial, env_steps=16)
    uninterrupted_state, uninterrupted_rng = agent.state_dict(), torch.get_rng_state().clone()
    restored, restored_buffer = PPOAgent(config), VectorRolloutBuffer(4, 2)
    restored.load_state_dict(saved_agent)
    restored_buffer.load_state_dict(saved_buffer)
    torch.set_rng_state(saved_rng)
    for index in range(2, 4):
        append_transition(restored_buffer, restored, observations[index], observations[index + 1])
    restored_buffer.compute_returns(.99, .95)
    resumed_metrics = restored.update(restored_buffer, env_steps=16)
    equal(restored_buffer.state_dict(), partial.state_dict())
    equal(restored.state_dict(), uninterrupted_state)
    equal(resumed_metrics, uninterrupted_metrics)
    assert torch.equal(torch.get_rng_state(), uninterrupted_rng)


@pytest.mark.parametrize("key,value", [
    ("target_kl", 0), ("target_kl", float("inf")), ("minibatch_size", 0),
    ("update_epochs", True), ("value_clipping", True), ("value_normalization", True),
    ("adam_betas", [1, .9]), ("adam_epsilon", 0),
])
def test_invalid_ppo_settings_rejected(config, key, value):
    config["ppo"][key] = value
    with pytest.raises(ValueError):
        PPOAgent(config)


@pytest.mark.parametrize("shape", [(16,), (8, 15), (8, 16, 1)])
def test_observation_shape_rejected(config, shape):
    with pytest.raises(ValueError):
        PPOAgent(config).act(np.zeros(shape))


def test_nonfinite_observation_and_malformed_checkpoint_rejected(config):
    agent = PPOAgent(config)
    with pytest.raises(ValueError):
        agent.act(np.full((8, 16), np.nan))
    with pytest.raises(ValueError):
        agent.load_state_dict(dict(format_version=0))
    buffer = rollout(agent)
    malformed = buffer.state_dict()
    malformed["arrays"]["observations"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        VectorRolloutBuffer(4, 8).load_state_dict(malformed)
    malformed_agent = agent.state_dict()
    malformed_agent["actor"]["log_std"][0] = 1
    with pytest.raises(ValueError):
        agent.load_state_dict(malformed_agent)
