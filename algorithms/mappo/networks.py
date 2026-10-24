"""Shared decentralized actor and team centralized critic."""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def _mlp(input_dim, hidden_sizes, output_dim, hidden_gain, output_gain):
    if not hidden_sizes or any(isinstance(size, bool) or not isinstance(size, int)
                               or size <= 0 for size in hidden_sizes):
        raise ValueError("hidden_sizes must be a nonempty list of positive integers")
    layers = []
    for size in hidden_sizes:
        layer = nn.Linear(input_dim, size)
        nn.init.orthogonal_(layer.weight, gain=hidden_gain)
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
        input_dim = size
    output = nn.Linear(input_dim, output_dim)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    return nn.Sequential(*layers, output)


def _input_scale(scale, dimension):
    scale = np.ones(dimension) if scale is None else np.asarray(scale)
    if scale.shape != (dimension,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("input_scale must have one finite positive value per input")
    return torch.tensor(scale, dtype=torch.float32)


class SharedActor(nn.Module):
    """Independent Gaussian latent actions followed by an elementwise tanh.

    Store latent (pre-tanh) actions for PPO. Reconstructing them with atanh from
    float32 saturated actions would lose information and can produce Inf.
    Log probabilities sum the three action components for each pursuer; the
    three pursuers' probabilities remain separate in the MAPPO surrogate.
    """

    def __init__(self, observation_dim, action_dim, hidden_sizes, initial_log_std,
                 min_log_std, max_log_std, hidden_gain, output_gain, input_scale=None):
        super().__init__()
        if not (np.isfinite([initial_log_std, min_log_std, max_log_std]).all()
                and min_log_std <= initial_log_std <= max_log_std
                and min_log_std < max_log_std):
            raise ValueError("Log standard deviation bounds must enclose the initial value")
        self.register_buffer("input_scale", _input_scale(input_scale, observation_dim))
        self.mean = _mlp(observation_dim, hidden_sizes, action_dim, hidden_gain, output_gain)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

    def forward(self, observations):
        return self.mean(observations / self.input_scale)

    def distribution(self, observations):
        mean = self(observations)
        std = self.log_std.clamp(self.min_log_std, self.max_log_std).exp()
        return Normal(mean, std.expand_as(mean))

    @staticmethod
    def log_prob_from_raw(distribution, raw_actions):
        """Exact change-of-variables correction with a stable log(1-tanh²)."""
        log_jacobian = 2.0 * (math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions))
        return (distribution.log_prob(raw_actions) - log_jacobian).sum(dim=-1)

    def sample(self, observations, deterministic=False):
        distribution = self.distribution(observations)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        return (raw_actions.tanh(), self.log_prob_from_raw(distribution, raw_actions), raw_actions)

    def evaluate_actions(self, observations, raw_actions):
        """Return old-action log density and a fresh entropy Monte Carlo estimate.

        The transformed distribution has no simple analytic entropy. A fresh
        reparameterized sample estimates its differential entropy, including
        the tanh Jacobian, with gradients through the current policy. Entropy
        can be negative; using the unsquashed Normal entropy would be wrong.
        """
        distribution = self.distribution(observations)
        log_prob = self.log_prob_from_raw(distribution, raw_actions)
        entropy = -self.log_prob_from_raw(distribution, distribution.rsample())
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """One scalar team value per complete 24D global state."""

    def __init__(self, state_dim, hidden_sizes, hidden_gain, output_gain, input_scale=None):
        super().__init__()
        self.register_buffer("input_scale", _input_scale(input_scale, state_dim))
        self.value = _mlp(state_dim, hidden_sizes, 1, hidden_gain, output_gain)

    def forward(self, states):
        return self.value(states / self.input_scale).squeeze(-1)
