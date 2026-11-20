"""Single-agent clipped PPO with fixed-scale observations and analytic KL."""

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn

from algorithms.mappo.networks import SharedActor, CentralizedCritic


def diagonal_gaussian_kl(old_means, old_log_stds, new_means, new_log_stds):
    """D_KL(old || new), summed over action dimensions for each observation.

    Both policies use the same invertible tanh transform. Their KL therefore
    equals this latent Gaussian KL. The old parameters come from collection.
    """
    if not (old_means.shape == old_log_stds.shape == new_means.shape == new_log_stds.shape):
        raise ValueError("Gaussian parameter shapes must match")
    log_ratio = new_log_stds - old_log_stds
    terms = log_ratio + .5 * ((-2 * log_ratio).exp()
                             + (old_means - new_means).square() * (-2 * new_log_stds).exp() - 1)
    return terms.sum(dim=-1).clamp_min(0)


class LinearEnvironmentScheduler:
    """Two learning rates determined by aggregate environment interactions."""

    def __init__(self, actor_optimizer, critic_optimizer, actor_start, actor_end,
                 critic_start, critic_end, total_env_steps, maximum_env_steps=None):
        rates = (actor_start, actor_end, critic_start, critic_end)
        if not np.isfinite(rates).all() or any(rate <= 0 for rate in rates):
            raise ValueError("Learning rate endpoints must be finite and positive")
        if isinstance(total_env_steps, bool) or not isinstance(total_env_steps, int) or total_env_steps <= 0:
            raise ValueError("Total environment budget must be a positive integer")
        maximum_env_steps = total_env_steps if maximum_env_steps is None else maximum_env_steps
        if (isinstance(maximum_env_steps, bool) or not isinstance(maximum_env_steps, int)
                or maximum_env_steps < total_env_steps):
            raise ValueError("Maximum environment budget must enclose the learning-rate decay horizon")
        self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer
        self.actor_start, self.actor_end = float(actor_start), float(actor_end)
        self.critic_start, self.critic_end = float(critic_start), float(critic_end)
        self.total_env_steps = total_env_steps
        self.maximum_env_steps = maximum_env_steps
        self.environment_steps = 0
        self.step(0)

    def step(self, environment_steps):
        if (isinstance(environment_steps, bool) or not isinstance(environment_steps, int)
                or not self.environment_steps <= environment_steps <= self.maximum_env_steps):
            raise ValueError("Scheduler environment counter must be monotonic and within budget")
        self.environment_steps = environment_steps
        fraction = environment_steps / self.total_env_steps
        if environment_steps >= self.total_env_steps:
            actor_lr, critic_lr = self.actor_end, self.critic_end
        else:
            actor_lr = self.actor_start + fraction * (self.actor_end - self.actor_start)
            critic_lr = self.critic_start + fraction * (self.critic_end - self.critic_start)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_lr
        return actor_lr, critic_lr

    def state_dict(self):
        state = dict(format_version=1, actor_start=self.actor_start, actor_end=self.actor_end,
                     critic_start=self.critic_start, critic_end=self.critic_end,
                     total_env_steps=self.total_env_steps, environment_steps=self.environment_steps)
        if self.maximum_env_steps != self.total_env_steps:
            state["maximum_env_steps"] = self.maximum_env_steps
        return state

    def load_state_dict(self, state, allow_budget_extension=False):
        for key in ("format_version", "actor_start", "actor_end", "critic_start", "critic_end", "total_env_steps"):
            if state.get(key) != self.state_dict()[key]:
                raise ValueError("Learning rate scheduler checkpoint differs: " + key)
        previous_maximum = state.get("maximum_env_steps", state["total_env_steps"])
        if (isinstance(previous_maximum, bool) or not isinstance(previous_maximum, int)
                or previous_maximum < state["total_env_steps"]):
            raise ValueError("Invalid checkpoint maximum environment budget")
        if previous_maximum != self.maximum_env_steps:
            if not allow_budget_extension or previous_maximum >= self.maximum_env_steps:
                raise ValueError("Learning rate scheduler checkpoint differs: maximum_env_steps")
        if (isinstance(state["environment_steps"], bool) or not isinstance(state["environment_steps"], int)
                or not 0 <= state["environment_steps"] <= previous_maximum):
            raise ValueError("Invalid checkpoint scheduler environment counter")
        self.environment_steps = 0
        self.step(state["environment_steps"])


class PPOAgent:
    """Independent actor and critic consume the same normalized 16-vector.

    Environment normalization occurs once, before this API. The reused network
    buffers consequently hold unit input scales. Global RNG belongs to the
    trainer checkpoint, allowing loading to be followed by exact RNG restore.
    """

    observation_dim = 16
    action_dim = 3

    def __init__(self, config):
        self.config = deepcopy(config)
        self.device = torch.device(config["device"])
        network, ppo = config["network"], config["ppo"]
        self.ppo = deepcopy(ppo)
        self._validate()
        initial_log_std = math.log(float(network["initial_std"]))
        self.actor = SharedActor(self.observation_dim, self.action_dim,
            network["actor_hidden_sizes"], initial_log_std, network["min_log_std"],
            network["max_log_std"], network["hidden_gain"], network["actor_output_gain"]).to(self.device)
        self.critic = CentralizedCritic(self.observation_dim, network["critic_hidden_sizes"],
            network["hidden_gain"], network["critic_output_gain"]).to(self.device)
        optimizer = dict(betas=tuple(ppo["adam_betas"]), eps=ppo["adam_epsilon"],
                         weight_decay=ppo["weight_decay"], amsgrad=ppo["amsgrad"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ppo["actor_learning_rate"], **optimizer)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=ppo["critic_learning_rate"], **optimizer)
        self.scheduler = LinearEnvironmentScheduler(self.actor_optimizer, self.critic_optimizer,
            ppo["actor_learning_rate"], ppo["actor_final_learning_rate"],
            ppo["critic_learning_rate"], ppo["critic_final_learning_rate"],
            config["training"].get("learning_rate_decay_steps", config["training"]["total_env_steps"]),
            maximum_env_steps=config["training"]["total_env_steps"])
        self.update_count = 0

    def _validate(self):
        ppo, network = self.ppo, self.config["network"]
        if ppo["value_clipping"] is not False or ppo["value_normalization"] is not False:
            raise ValueError("This PPO protocol disables value clipping and value normalization")
        for key in ("adam_epsilon", "max_grad_norm", "normalization_epsilon", "target_kl"):
            value = ppo[key]
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("Invalid positive PPO value: " + key)
        for key in ("entropy_coefficient", "value_loss_coefficient", "weight_decay"):
            if not np.isfinite(ppo[key]) or ppo[key] < 0:
                raise ValueError("Invalid nonnegative PPO value: " + key)
        for key in ("gamma", "gae_lambda"):
            if not np.isfinite(ppo[key]) or not 0 <= ppo[key] <= 1:
                raise ValueError("Invalid discount/GAE value: " + key)
        if not 0 < ppo["clip_ratio"] < 1:
            raise ValueError("clip_ratio must be within (0,1)")
        for key in ("minibatch_size", "update_epochs"):
            if isinstance(ppo[key], bool) or not isinstance(ppo[key], int) or ppo[key] <= 0:
                raise ValueError(key + " must be a positive integer")
        betas = ppo["adam_betas"]
        if len(betas) != 2 or not np.isfinite(betas).all() or not all(0 <= b < 1 for b in betas):
            raise ValueError("Adam betas must contain two values in [0,1)")
        if not isinstance(ppo["amsgrad"], bool) or not isinstance(ppo["advantage_normalization"], bool):
            raise ValueError("amsgrad and advantage_normalization must be boolean")
        if not np.isfinite(network["initial_std"]) or network["initial_std"] <= 0:
            raise ValueError("initial_std must be finite and positive")

    def _tensor(self, array):
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        if not torch.isfinite(tensor).all():
            raise ValueError("Network inputs must be finite")
        return tensor

    def _observations(self, observations):
        tensor = self._tensor(observations)
        if tensor.ndim != 2 or tensor.shape[-1] != self.observation_dim:
            raise ValueError("Observations must have shape (num_environments,16)")
        return tensor

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        observations = self._observations(observations)
        distribution = self.actor.distribution(observations)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        data = dict(actions=raw_actions.tanh(), raw_actions=raw_actions,
                    log_probs=self.actor.log_prob_from_raw(distribution, raw_actions),
                    means=distribution.mean,
                    log_stds=self.actor.log_std.clamp(self.actor.min_log_std, self.actor.max_log_std).expand_as(raw_actions),
                    values=self.critic(observations))
        if not all(torch.isfinite(value).all() for value in data.values()):
            raise FloatingPointError("Non-finite action or value")
        return {key: value.cpu().numpy().copy() for key, value in data.items()}

    @torch.no_grad()
    def value(self, observations):
        values = self.critic(self._observations(observations))
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite critic prediction")
        return values.cpu().numpy().copy()

    def update(self, buffer, env_steps):
        if buffer.observation_dim != self.observation_dim or buffer.action_dim != self.action_dim:
            raise ValueError("Rollout dimensions differ from this PPO policy")
        flat = buffer.flattened()
        actor_lr, critic_lr = self.scheduler.step(env_steps)
        tensors = {name: self._tensor(flat[name]) for name in (
            "observations", "raw_actions", "log_probs", "means", "log_stds", "advantages", "returns")}
        advantages = tensors["advantages"]
        if self.ppo["advantage_normalization"]:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + self.ppo["normalization_epsilon"])
        count = len(advantages)
        self.actor.train()
        self.critic.train()
        minibatches, epochs = [], []
        total_weight = 0
        fields = ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                  "actor_grad_norm", "critic_grad_norm")
        totals = dict.fromkeys(fields, 0.0)
        early_stopped = False
        for epoch_index in range(self.ppo["update_epochs"]):
            permutation = torch.randperm(count, device=self.device)
            epoch_rows = []
            for start in range(0, count, self.ppo["minibatch_size"]):
                indices = permutation[start:start + self.ppo["minibatch_size"]]
                observations, raw = tensors["observations"][indices], tensors["raw_actions"][indices]
                old_log_probs = tensors["log_probs"][indices]
                log_probs, entropy = self.actor.evaluate_actions(observations, raw)
                ratio = (log_probs - old_log_probs).exp()
                batch_advantages = advantages[indices]
                actor_loss = -torch.minimum(ratio * batch_advantages,
                    ratio.clamp(1 - self.ppo["clip_ratio"], 1 + self.ppo["clip_ratio"]) * batch_advantages).mean()
                entropy_mean = entropy.mean()
                policy_objective = actor_loss - self.ppo["entropy_coefficient"] * entropy_mean
                values = self.critic(observations)
                value_loss = .5 * (values - tensors["returns"][indices]).square().mean()
                value_objective = self.ppo["value_loss_coefficient"] * value_loss
                if not all(torch.isfinite(v).all() for v in (ratio, policy_objective, value_objective)):
                    raise FloatingPointError("Non-finite PPO objective")
                self.actor_optimizer.zero_grad(set_to_none=True)
                policy_objective.backward()
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.ppo["max_grad_norm"], error_if_nonfinite=True)
                self.actor_optimizer.step()
                with torch.no_grad():
                    self.actor.log_std.clamp_(self.actor.min_log_std, self.actor.max_log_std)
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_objective.backward()
                critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.ppo["max_grad_norm"], error_if_nonfinite=True)
                self.critic_optimizer.step()
                with torch.no_grad():
                    current = self.actor.distribution(observations)
                    current_logs = self.actor.log_std.clamp(self.actor.min_log_std, self.actor.max_log_std).expand_as(current.mean)
                    kl = diagonal_gaussian_kl(tensors["means"][indices], tensors["log_stds"][indices], current.mean, current_logs).mean()
                    post_log_probs = self.actor.log_prob_from_raw(current, raw)
                    post_ratio = (post_log_probs - old_log_probs).exp()
                    clip_fraction = ((post_ratio - 1).abs() > self.ppo["clip_ratio"]).float().mean()
                    if not torch.isfinite(kl) or not torch.isfinite(post_ratio).all():
                        raise FloatingPointError("Non-finite post-update policy diagnostic")
                record = dict(epoch=epoch_index + 1, minibatch=len(epoch_rows) + 1, samples=len(indices),
                              actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()),
                              entropy=float(entropy_mean.detach()), approx_kl=float(kl),
                              analytic_kl=float(kl), clip_fraction=float(clip_fraction),
                              actor_grad_norm=float(actor_grad), critic_grad_norm=float(critic_grad),
                              actor_learning_rate=actor_lr, critic_learning_rate=critic_lr,
                              early_stop_triggered=bool(kl > self.ppo["target_kl"]))
                minibatches.append(record)
                epoch_rows.append(record)
                total_weight += len(indices)
                for name in fields:
                    totals[name] += record[name] * len(indices)
                if record["early_stop_triggered"]:
                    early_stopped = True
                    break
            epoch_samples = sum(row["samples"] for row in epoch_rows)
            epochs.append(dict(epoch=epoch_index + 1, minibatches=len(epoch_rows), samples=epoch_samples,
                               clip_fraction=sum(r["clip_fraction"] * r["samples"] for r in epoch_rows) / epoch_samples,
                               approx_kl=sum(r["approx_kl"] * r["samples"] for r in epoch_rows) / epoch_samples,
                               early_stopped=early_stopped))
            if early_stopped:
                break
        self.update_count += 1
        result = {name: value / total_weight for name, value in totals.items()}
        result.update(actor_learning_rate=actor_lr, critic_learning_rate=critic_lr,
                      updates_completed_minibatches=len(minibatches), epochs_completed=len(epochs),
                      early_stopped=early_stopped, last_kl=minibatches[-1]["approx_kl"],
                      max_kl=max(row["approx_kl"] for row in minibatches),
                      early_stop_kl=minibatches[-1]["approx_kl"] if early_stopped else None,
                      log_std=self.actor.log_std.detach().cpu().tolist(),
                      std=self.actor.log_std.detach().exp().cpu().tolist(),
                      minibatches=minibatches, epochs=epochs)
        return result

    def state_dict(self):
        return dict(format_version=1, config=deepcopy(self.config),
                    actor=deepcopy(self.actor.state_dict()), critic=deepcopy(self.critic.state_dict()),
                    actor_optimizer=deepcopy(self.actor_optimizer.state_dict()),
                    critic_optimizer=deepcopy(self.critic_optimizer.state_dict()),
                    scheduler=self.scheduler.state_dict(), update_count=self.update_count)

    def load_state_dict(self, state, allow_budget_extension=False):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported single-agent PPO checkpoint")
        for key in ("network", "ppo"):
            if state["config"][key] != self.config[key]:
                raise ValueError("Checkpoint learning configuration differs: " + key)
        count = state.get("update_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid checkpoint update counter")
        for network in ("actor", "critic"):
            if any(not torch.isfinite(value).all() for value in state[network].values()):
                raise ValueError("Non-finite checkpoint network: " + network)
        logs = state["actor"]["log_std"]
        if not bool(((logs >= self.actor.min_log_std) & (logs <= self.actor.max_log_std)).all()):
            raise ValueError("Checkpoint log_std is outside the parameter bounds")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.scheduler.load_state_dict(state["scheduler"], allow_budget_extension=allow_budget_extension)
        self.update_count = state["update_count"]
        self.actor.log_std.requires_grad_(True)
