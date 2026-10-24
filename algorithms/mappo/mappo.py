"""Shared-policy MAPPO with one centralized team value function."""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .networks import CentralizedCritic, SharedActor


class MAPPOAgent:
    """Per-agent PPO surrogates share team GAE; the critic is trained once.

    Minibatches select team timesteps. A minibatch of B steps yields B*3 actor
    samples and B critic samples. The policy only consumes local observations;
    the centralized state never enters the actor.
    """

    def __init__(self, config, observation_dim=21, state_dim=24, action_dim=3,
                 observation_scale=None, state_scale=None, device=None):
        self.config = deepcopy(config)
        self.device = torch.device(config["device"] if device is None else device)
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ppo = self.config["ppo"]
        self._validate_ppo()
        network = config["network"]
        self.actor = SharedActor(
            observation_dim, action_dim, network["actor_hidden_sizes"],
            network["initial_log_std"], network["min_log_std"], network["max_log_std"],
            network["hidden_gain"], network["actor_output_gain"], observation_scale,
        ).to(self.device)
        self.critic = CentralizedCritic(
            state_dim, network["critic_hidden_sizes"], network["hidden_gain"],
            network["critic_output_gain"], state_scale,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.ppo["actor_learning_rate"], eps=self.ppo["adam_epsilon"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.ppo["critic_learning_rate"], eps=self.ppo["adam_epsilon"]
        )

    def _validate_ppo(self):
        positive = ("actor_learning_rate", "critic_learning_rate", "adam_epsilon",
                    "max_grad_norm", "normalization_epsilon")
        nonnegative = ("entropy_coefficient", "value_loss_coefficient")
        for key in positive + nonnegative:
            value = self.ppo[key]
            if not np.isfinite(value) or (value <= 0 if key in positive else value < 0):
                raise ValueError("Invalid positive/nonnegative PPO parameter: {}".format(key))
        for key in ("gamma", "gae_lambda"):
            if not np.isfinite(self.ppo[key]) or not 0 <= self.ppo[key] <= 1:
                raise ValueError("{} must lie in [0, 1]".format(key))
        if not np.isfinite(self.ppo["clip_ratio"]) or not 0 < self.ppo["clip_ratio"] < 1:
            raise ValueError("clip_ratio must lie in (0, 1)")
        for key in ("update_epochs", "minibatch_size"):
            value = self.ppo[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(key))
        if not isinstance(self.ppo["advantage_normalization"], bool):
            raise ValueError("advantage_normalization must be a boolean")

    def _tensor(self, array):
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        if not torch.isfinite(tensor).all():
            raise ValueError("Network inputs must be finite")
        return tensor

    @torch.no_grad()
    def act(self, observations, global_state=None, deterministic=False):
        """Return bounded actions, log probabilities, latent actions, team value.

        Omitting global_state executes only the decentralized actor, as used
        in evaluation. The latent actions are for training storage, not env.step.
        """
        observations = self._tensor(observations)
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError("observations must have shape (num_agents, observation_dim)")
        actions, log_probs, raw_actions = self.actor.sample(observations, deterministic)
        if not all(torch.isfinite(result).all() for result in (actions, log_probs, raw_actions)):
            raise FloatingPointError("Non-finite actor output")
        value = None if global_state is None else self.value(global_state)
        return (actions.cpu().numpy(), log_probs.cpu().numpy(), raw_actions.cpu().numpy(), value)

    @torch.no_grad()
    def value(self, global_state):
        state = self._tensor(global_state)
        if state.shape != (self.state_dim,):
            raise ValueError("global_state must have shape (state_dim,)")
        value = self.critic(state)
        if not torch.isfinite(value):
            raise FloatingPointError("Non-finite critic output")
        return float(value.item())

    def update(self, buffer):
        if buffer.size == 0 or not buffer.returns_ready:
            raise RuntimeError("Compute GAE on a nonempty rollout before updating")
        size = buffer.size
        observations = self._tensor(buffer.observations[:size])
        states = self._tensor(buffer.states[:size])
        raw_actions = self._tensor(buffer.raw_actions[:size])
        old_log_probs = self._tensor(buffer.log_probs[:size])
        returns = self._tensor(buffer.returns[:size])
        advantages = self._tensor(buffer.advantages[:size])
        if self.ppo["advantage_normalization"]:
            # Population std also behaves correctly for a one-timestep batch.
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + self.ppo["normalization_epsilon"]
            )
        totals = dict.fromkeys(("actor_loss", "value_loss", "entropy", "approx_kl",
                                "clip_fraction", "actor_grad_norm", "critic_grad_norm"), 0.0)
        weight = 0
        self.actor.train()
        self.critic.train()
        for _ in range(self.ppo["update_epochs"]):
            indices = torch.randperm(size, device=self.device)
            for start in range(0, size, self.ppo["minibatch_size"]):
                batch = indices[start:start + self.ppo["minibatch_size"]]
                batch_observations = observations[batch].reshape(-1, self.observation_dim)
                batch_raw_actions = raw_actions[batch].reshape(-1, self.action_dim)
                log_probs, entropy = self.actor.evaluate_actions(batch_observations, batch_raw_actions)
                log_ratio = log_probs - old_log_probs[batch].reshape(-1)
                ratio = log_ratio.exp()
                team_advantages = advantages[batch, None].expand(-1, buffer.num_agents).reshape(-1)
                unclipped = ratio * team_advantages
                clipped = ratio.clamp(1 - self.ppo["clip_ratio"], 1 + self.ppo["clip_ratio"]) * team_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                entropy_mean = entropy.mean()
                policy_objective = actor_loss - self.ppo["entropy_coefficient"] * entropy_mean
                values = self.critic(states[batch])
                value_loss = 0.5 * (values - returns[batch]).square().mean()
                value_objective = self.ppo["value_loss_coefficient"] * value_loss
                if not all(torch.isfinite(result).all() for result in
                           (ratio, policy_objective, value_objective)):
                    raise FloatingPointError("Non-finite PPO objective")

                self.actor_optimizer.zero_grad(set_to_none=True)
                policy_objective.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.ppo["max_grad_norm"], error_if_nonfinite=True
                )
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_objective.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.ppo["max_grad_norm"], error_if_nonfinite=True
                )
                self.critic_optimizer.step()

                with torch.no_grad():
                    metrics = {
                        "actor_loss": actor_loss, "value_loss": value_loss, "entropy": entropy_mean,
                        "approx_kl": ((ratio - 1) - log_ratio).mean(),
                        "clip_fraction": ((ratio - 1).abs() > self.ppo["clip_ratio"]).float().mean(),
                        "actor_grad_norm": actor_grad_norm, "critic_grad_norm": critic_grad_norm,
                    }
                    for key, value in metrics.items():
                        totals[key] += float(value.item()) * len(batch)
                    weight += len(batch)
        return {key: value / weight for key, value in totals.items()}

    def save(self, path, training_state=None, metadata=None):
        """Persist model, optimizer, configuration, provenance and random states."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "config": self.config,
            "dimensions": {"observation_dim": self.observation_dim, "state_dim": self.state_dim,
                           "action_dim": self.action_dim},
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_state": {} if training_state is None else training_state,
            "metadata": {} if metadata is None else metadata,
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        # Replace only after serialization succeeds; avoid a partially written final.pt.
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        # This project checkpoint includes configs/NumPy RNG state, so request
        # full loading explicitly rather than depending on torch's version default.
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported MAPPO checkpoint format")
        agent = cls(payload["config"], device=device, **payload["dimensions"])
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        agent.actor.eval()
        agent.critic.eval()
        # RNG restoration is intentionally left to a future resume-training
        # runner. Loading a policy for evaluation does not restore training RNG.
        return agent, payload
